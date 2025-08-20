# 🤖 M.H.Z Library Agent Tests
# 👨‍💻 Author: Muhammad Hammad Zubair

import asyncio
from rich.console import Console
from rich.panel import Panel

# Import from your main library file
from main_libraray import library_agent, config, UserContext, Runner

console = Console()

async def run_tests():
    # Fixed user context (registered member)
    context = UserContext(name="Ali", member_id="M001")

    # Required 3 queries
    queries = [
        "Search for Python Crash Course",
        "Check availability of Clean Code",
        "What are library timings?",
    ]

    console.print(Panel("📚 [bold cyan]M.H.Z Library Agent - Automated Tests[/]", subtitle="Assignment Mode", style="bright_blue"))

    for i, q in enumerate(queries, start=1):
        try:
            result = await Runner.run(library_agent, q, context=context, run_config=config)
            console.print(
                Panel(
                    f"❓ [bold yellow]Query {i}:[/] {q}\n\n🤖 [bold green]Answer:[/] {result.final_output}",
                    style="cyan",
                )
            )
        except Exception as e:
            console.print(Panel(f"⚠️ Error for query '{q}': {e}", style="red"))

if __name__ == "__main__":
    asyncio.run(run_tests())
