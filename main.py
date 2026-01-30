#!/usr/bin/env python3
"""RAG Application CLI - Chat with your documents using AI."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def validate_folder(ctx, param, value):
    """Validate that the folder exists."""
    folder = Path(value).resolve()
    if not folder.exists():
        raise click.BadParameter(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise click.BadParameter(f"Not a directory: {folder}")
    return str(folder)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RAG Application - Chat with your documents using AI.

    Index a folder of documents and ask questions about them using
    your choice of LLM (Anthropic Claude or OpenAI GPT).
    """
    pass


@cli.command()
@click.argument("folder", callback=validate_folder)
@click.option(
    "--embedding", "-e",
    type=click.Choice(["openai", "local"]),
    default=None,
    help="Embedding provider (default: from config or 'local')"
)
@click.option(
    "--rebuild", "-r",
    is_flag=True,
    help="Force rebuild the index"
)
def index(folder: str, embedding: str, rebuild: bool):
    """Index documents in a folder.

    FOLDER is the path to the directory containing documents to index.
    Supports: .txt, .md, code files, .pdf, .docx
    """
    from rag_engine import RAGEngine

    console.print(f"\n[bold blue]Indexing folder:[/bold blue] {folder}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing...", total=None)
            engine = RAGEngine(folder, embedding_provider=embedding)

        if engine.is_indexed() and not rebuild:
            count = engine.document_count()
            console.print(f"[yellow]Index already exists with {count} chunks.[/yellow]")
            if not click.confirm("Rebuild index?"):
                return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading and indexing documents...", total=None)
            chunk_count = engine.index_documents(rebuild=rebuild)

        if chunk_count > 0:
            console.print(f"\n[green]Successfully indexed {chunk_count} document chunks.[/green]")
        else:
            console.print("\n[yellow]No documents found to index.[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("folder", callback=validate_folder)
@click.argument("question")
@click.option(
    "--llm", "-l",
    type=click.Choice(["anthropic", "openai"]),
    default=None,
    help="LLM provider (default: from config or 'anthropic')"
)
@click.option(
    "--embedding", "-e",
    type=click.Choice(["openai", "local"]),
    default=None,
    help="Embedding provider (default: from config or 'local')"
)
def query(folder: str, question: str, llm: str, embedding: str):
    """Ask a single question about the indexed documents.

    FOLDER is the path to the indexed directory.
    QUESTION is your question about the documents.
    """
    from rag_engine import RAGEngine

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing...", total=None)
            engine = RAGEngine(folder, llm_provider=llm, embedding_provider=embedding)

        if not engine.is_indexed():
            console.print("[red]Error:[/red] Folder is not indexed. Run 'index' command first.")
            sys.exit(1)

        console.print(f"\n[bold]Using:[/bold] {engine.llm_service.name}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Thinking...", total=None)
            response = engine.query(question, stream=False)

        console.print(Panel(Markdown(response), title="Answer", border_style="green"))

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("folder", callback=validate_folder)
@click.option(
    "--llm", "-l",
    type=click.Choice(["anthropic", "openai"]),
    default=None,
    help="LLM provider (default: from config or 'anthropic')"
)
@click.option(
    "--embedding", "-e",
    type=click.Choice(["openai", "local"]),
    default=None,
    help="Embedding provider (default: from config or 'local')"
)
def chat(folder: str, llm: str, embedding: str):
    """Start an interactive chat session with the indexed documents.

    FOLDER is the path to the indexed directory.

    Type 'quit' or 'exit' to end the session.
    Type 'switch' to change LLM provider mid-session.
    """
    from rag_engine import RAGEngine

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing...", total=None)
            engine = RAGEngine(folder, llm_provider=llm, embedding_provider=embedding)

        if not engine.is_indexed():
            console.print("[red]Error:[/red] Folder is not indexed. Run 'index' command first.")
            sys.exit(1)

        # Check for instructions
        if engine.instructions:
            console.print("[dim]Custom instructions loaded from instructions.txt[/dim]")

        console.print(Panel(
            f"[bold]RAG Chat Session[/bold]\n\n"
            f"Folder: {folder}\n"
            f"Documents: {engine.document_count()} chunks\n"
            f"LLM: {engine.llm_service.name}\n\n"
            f"[dim]Commands: 'quit' to exit, 'switch' to change LLM[/dim]",
            border_style="blue"
        ))

        while True:
            try:
                console.print()
                question = console.input("[bold cyan]You:[/bold cyan] ").strip()

                if not question:
                    continue

                if question.lower() in ("quit", "exit", "q"):
                    console.print("\n[dim]Goodbye![/dim]")
                    break

                if question.lower() == "switch":
                    current = "anthropic" if "anthropic" in engine.llm_service.name.lower() else "openai"
                    new_provider = "openai" if current == "anthropic" else "anthropic"
                    try:
                        from llm_service import get_llm_service
                        engine.llm_service = get_llm_service(new_provider)
                        console.print(f"[green]Switched to {engine.llm_service.name}[/green]")
                    except Exception as e:
                        console.print(f"[red]Could not switch: {e}[/red]")
                    continue

                # Stream the response
                console.print("\n[bold green]Assistant:[/bold green] ", end="")
                for chunk in engine.query(question, stream=True):
                    console.print(chunk, end="")
                console.print()

            except KeyboardInterrupt:
                console.print("\n\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
                continue

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("folder", callback=validate_folder)
def status(folder: str):
    """Show the index status for a folder."""
    from pathlib import Path
    from config import Config

    index_path = Path(folder) / Config.INDEX_FOLDER
    instructions_path = Path(folder) / Config.INSTRUCTIONS_FILE

    console.print(f"\n[bold]Folder:[/bold] {folder}")

    if index_path.exists():
        try:
            from embeddings import get_embedding_service
            from vector_store import VectorStore
            embedding_service = get_embedding_service("local")
            store = VectorStore(folder, embedding_service)
            count = store.count()
            console.print(f"[green]Index exists:[/green] {count} document chunks")
        except Exception as e:
            console.print(f"[yellow]Index exists but could not read: {e}[/yellow]")
    else:
        console.print("[yellow]Not indexed[/yellow]")

    if instructions_path.exists():
        console.print(f"[green]Instructions file found[/green]")
    else:
        console.print("[dim]No instructions.txt[/dim]")

    console.print()


if __name__ == "__main__":
    cli()
