import os
import asyncio
from typing import Any
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    function_tool,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    ModelSettings,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from difflib import get_close_matches

# Windows async fix
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ------------------ Databases ------------------
class UserContext(BaseModel):
    name: str
    member_id: str | None = None

BOOK_DB: dict[str, int] = {
    "clean code": 3,
    "python crash course": 2,
    "deep learning": 1,
    "effective java": 2,
    "learning python": 4,
}
MEMBERS_DB: dict[str, str] = {
    "1111": "Tehreem",
    "2222": "Dua",
    "3333": "Nashrah",
}

# ------------------ Guardrail ------------------
class LibraryGuardOutput(BaseModel):
    is_library_question: bool
    reason: str

guardrail_agent = Agent(
    name="LibraryTopicGuard",
    instructions=(
        "Mark as library-related if user asks about: books, availability, timings, "
        "or says a book name directly. Be lenient, don't block simple book titles."
    ),
    output_type=LibraryGuardOutput,
    model=model
)

@input_guardrail
async def library_input_guardrail(
    ctx: RunContextWrapper[UserContext],
    agent: Agent,
    input: str | list[Any],
) -> GuardrailFunctionOutput:
    user_text = input if isinstance(input, str) else " ".join(input)
    # Auto-pass if matches a book title or generic book words
    keywords = ["book", "kitab", "library", "timing", "available", "copies"]
    if any(k in user_text.lower() for k in keywords) or get_close_matches(user_text.lower(), BOOK_DB.keys()):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    result = await Runner.run(guardrail_agent, user_text, context=ctx.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.is_library_question),
    )

# ------------------ Tools ------------------
@function_tool
def list_books(wrapper: RunContextWrapper[UserContext]) -> dict:
    return {"available_books": [{title.title(): copies} for title, copies in BOOK_DB.items()]}

@function_tool
def search_book(wrapper: RunContextWrapper[UserContext], query: str) -> dict:
    query_lower = query.lower()
    matches = [t.title() for t in BOOK_DB.keys() if query_lower in t]
    if not matches:  # fuzzy match if exact not found
        close = get_close_matches(query_lower, BOOK_DB.keys(), n=3, cutoff=0.6)
        matches = [t.title() for t in close]
    return {"found": bool(matches), "matches": matches}

@function_tool
def check_availability(wrapper: RunContextWrapper[UserContext], title: str) -> dict:
    member_id = getattr(wrapper.context, "member_id", None)
    if not member_id or member_id not in MEMBERS_DB:
        return {"error": "Not a registered member. Register to check availability."}
    close = get_close_matches(title.lower(), BOOK_DB.keys(), n=1, cutoff=0.6)
    if not close:
        return {"title": title.title(), "copies": 0}
    title_match = close[0]
    copies = BOOK_DB.get(title_match, 0)
    return {"title": title_match.title(), "copies": copies}

@function_tool
def library_timings(wrapper: RunContextWrapper[UserContext]) -> str:
    return "Library timings: Mon-Fri 09:00-18:00, Sat 10:00-14:00, Sun Closed."

# --- New Tool: Intelligent Book Search ---
@function_tool
async def intelligent_book_search(wrapper: RunContextWrapper[UserContext], topic: str) -> dict:
    query_lower = topic.lower()
    
    # Step 1: Check local DB first
    matches = [t.title() for t in BOOK_DB.keys() if query_lower in t]
    if not matches:
        close = get_close_matches(query_lower, BOOK_DB.keys(), n=3, cutoff=0.6)
        matches = [t.title() for t in close]

    if matches:
        return {"source": "local", "matches": matches}

    # Step 2: Use LLM for suggestions
    prompt_text = f"Suggest 3 popular and useful books for learning about '{topic}'. Provide book titles and authors."
    result = await Runner.run(
        Agent(
            name="BookRecommender",
            instructions="You are a helpful AI that recommends books.",
            model=model,
            model_settings=ModelSettings(temperature=0.7)
        ),
        prompt_text,
        run_config=config
    )

    suggestions = result.final_output.strip()
    return {"source": "llm", "matches": suggestions.split("\n")}

# ------------------ Dynamic Instructions ------------------
def dynamic_instructions(context: RunContextWrapper[UserContext], agent: Agent) -> str:
    name = getattr(context.context, "name", "Guest")
    return (
        f"You are a helpful library assistant for {name}. "
        "You can: search books, check availability, list all books, and provide timings. "
        "If a book is not found locally, use your intelligence to suggest relevant books."
    )

# ------------------ Main Agent ------------------
library_agent = Agent[UserContext](
    name="LibraryAssistant",
    instructions=dynamic_instructions,
    tools=[list_books, search_book, check_availability, library_timings, intelligent_book_search],
    input_guardrails=[library_input_guardrail],
    model=model,
    model_settings=ModelSettings(temperature=0.0),
)

# ------------------ CLI ------------------
console = Console()

def main():
    console.print(Panel("📚 [bold cyan]T.KZ Intelligent Library Assistant[/]", subtitle="Made by TEHREEM", style="bright_blue"))
    name = prompt("👤 Name: ", history=InMemoryHistory())
    member_id = prompt("🆔 Member ID (or leave empty if not registered): ", history=InMemoryHistory()).strip() or None
    context = UserContext(name=name, member_id=member_id)

    console.print("\n[bold green]💬 Type your library query below. (type 'exit' to quit)[/]")
    while True:
        console.print("\n" + "-"*60, style="grey37")
        user_input = prompt("⚡ Query>>> ", history=InMemoryHistory())
        if user_input.strip().lower() == "exit":
            console.print(Panel("[bold magenta]✅ Thank you for using T.KZ Library Assistant. Goodbye![/]", style="magenta"))
            break
        console.print(Text("🧠 Processing query through T.KZ Intelligence...", style="bold yellow"))
        try:
            result = asyncio.run(Runner.run(library_agent, user_input, context=context, run_config=config))
            console.print(Panel(f"[bold white]{result.final_output}[/]", title="🤖 T.KZ Agent Reply", style="green"))
        except InputGuardrailTripwireTriggered:
            console.print(Panel("[bold red]❌ Not a library-related query. Please ask about books, availability, or timings.[/]", style="red"))
        except Exception as e:
            console.print(Panel(f"[bold red]Error: {e}[/]", style="red"))
    console.print("\n")

if __name__ == "__main__":
    main()

