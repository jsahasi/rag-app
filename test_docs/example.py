"""Example Python module for testing RAG indexing."""

class DataProcessor:
    """A class that processes various types of data."""

    def __init__(self, config: dict):
        """Initialize the processor with configuration.

        Args:
            config: Dictionary containing processor settings
        """
        self.config = config
        self.processed_count = 0

    def process_file(self, filepath: str) -> dict:
        """Process a single file and return results.

        Args:
            filepath: Path to the file to process

        Returns:
            Dictionary containing processing results
        """
        # Read the file
        with open(filepath, 'r') as f:
            content = f.read()

        # Process content
        result = {
            'filepath': filepath,
            'length': len(content),
            'lines': content.count('\n') + 1,
        }

        self.processed_count += 1
        return result

    def get_statistics(self) -> dict:
        """Get processing statistics.

        Returns:
            Dictionary with processing stats
        """
        return {
            'total_processed': self.processed_count,
            'config': self.config
        }


def main():
    """Main entry point for the processor."""
    config = {
        'max_file_size': 1024 * 1024,  # 1MB
        'allowed_extensions': ['.txt', '.md', '.json'],
    }

    processor = DataProcessor(config)

    # Example usage
    print("Data Processor initialized")
    print(f"Config: {processor.config}")


if __name__ == "__main__":
    main()
