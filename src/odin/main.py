"""Main entry point for ODIN TUI application."""
import logging
import sys

from odin.tui import run_tui


def main():
    """Main entry point."""
    # Configure logging to file only (avoid cluttering TUI)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("odin.log"),
        ],
    )
    
    try:
        run_tui()
    except KeyboardInterrupt:
        print("\nExiting ODIN...")
        sys.exit(0)
    except Exception as e:
        logging.exception("Fatal error")
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
