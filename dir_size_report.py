import os
import time
import ctypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import tkinter as tk
from tkinter import filedialog

class TimeoutManager:
    """Manage timeout state across recursive calls."""
    def __init__(self):
        self.timed_out = False
        self.start_time = time.time()
        self.timeout_seconds = 0

    def set_timeout(self, timeout_seconds: int):
        """Set timeout duration."""
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.timed_out = False

    def check_timeout(self) -> bool:
        """Check if timeout has been reached."""
        if not self.timed_out and (time.time() - self.start_time) >= self.timeout_seconds:
            self.timed_out = True
        return self.timed_out

timeout_manager = TimeoutManager()

def get_file_size(file_path: str, mode: str) -> int:
    """Get file size in bytes based on mode ('size' or 'size_on_disk')."""
    try:
        stat = os.stat(file_path)
        if mode == "size":
            return stat.st_size
        elif mode == "size_on_disk":
            if os.name == "nt":  # Windows
                # Use ctypes to get actual disk usage (handles OneDrive placeholder files)
                high_size = ctypes.c_ulonglong(0)
                low_size = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetCompressedFileSizeW(
                    ctypes.c_wchar_p(file_path), ctypes.byref(high_size), ctypes.byref(low_size)
                )
                return (high_size.value << 32) + low_size.value
            else:  # Unix-like
                # Use st_blocks * 512 (standard block size for disk usage)
                return stat.st_blocks * 512
    except (PermissionError, OSError):
        return 0  # Skip inaccessible files

def get_folder_size(folder_path: str, mode: str) -> int:
    """Calculate total size of folder contents in bytes based on mode."""
    if timeout_manager.check_timeout():
        return 0  # Stop size calculation if timed out
    total_size = 0
    try:
        for entry in os.scandir(folder_path):
            if entry.is_file():
                total_size += get_file_size(entry.path, mode)
            elif entry.is_dir():
                total_size += get_folder_size(entry.path, mode)
    except (PermissionError, OSError):
        pass  # Skip inaccessible folders
    return total_size

def format_size(size: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def crawl_directory(
    parent_dir: str,
    threshold_bytes: int,
    timeout_seconds: int,
    mode: str,
    level: int = 0
) -> None:
    """Crawl directory and print folders exceeding threshold with tabbed hierarchy."""
    if timeout_manager.check_timeout():
        return  # Stop crawling if timed out

    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        print(f"Error: {parent_dir} is not a valid directory")
        return

    # Get parent folder size
    parent_size = get_folder_size(parent_dir, mode)
    if timeout_manager.check_timeout():
        return  # Stop if timed out during size calculation

    # Print parent folder if it exceeds threshold or is the root
    if parent_size > threshold_bytes or level == 0:
        print(f"{'  ' * level}{parent_path.name} {format_size(parent_size)}")

    # Check subdirectories
    try:
        for entry in sorted(os.scandir(parent_dir), key=lambda x: x.name):
            if timeout_manager.check_timeout():
                return  # Stop if timed out
            if entry.is_dir():
                try:
                    folder_size = get_folder_size(entry.path, mode)
                    if timeout_manager.check_timeout():
                        return  # Stop if timed out during size calculation
                    # Recursively crawl subfolder if it exceeds threshold
                    if folder_size > threshold_bytes:
                        crawl_directory(
                            entry.path,
                            threshold_bytes,
                            timeout_seconds,
                            mode,
                            level + 1
                        )
                except (PermissionError, OSError):
                    continue
    except TimeoutError:
        timeout_manager.timed_out = True
        return

def run_with_timeout(parent_dir: str, threshold_bytes: int, timeout_seconds: int, mode: str):
    """Run crawling with timeout using ThreadPoolExecutor."""
    timeout_manager.set_timeout(timeout_seconds)
    
    def crawl_task():
        crawl_directory(parent_dir, threshold_bytes, timeout_seconds, mode)
        if timeout_manager.timed_out:
            print("Crawling stopped due to timeout")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(crawl_task)
        try:
            future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            timeout_manager.timed_out = True
            print("Crawling stopped due to timeout")

def select_directory():
    """Open a file dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title="Select Parent Directory")
    root.destroy()
    return directory

def main():
    # Select parent directory
    print("Please select the parent directory in the dialog window...")
    parent_dir = select_directory()
    if not parent_dir:
        print("Error: No directory selected")
        return

    # Get user input with defaults
    try:
        threshold_input = input("Enter size threshold in GB (2): ").strip()
        threshold = float(threshold_input) if threshold_input else 2.0
        threshold_bytes = int(threshold * 1024 ** 3)  # Convert GB to bytes

        timeout_input = input("Enter timeout in seconds (60): ").strip()
        timeout = int(timeout_input) if timeout_input else 60

        mode_input = input("Enter mode ('size' or 'size_on_disk') (size): ").strip().lower()
        mode = mode_input if mode_input in ["size", "size_on_disk"] else "size"
    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return

    # Run crawler
    print(f"\nCrawling {parent_dir} for folders larger than {threshold} GB in {mode} mode")
    print("Format: Folder Name Size")
    print("-" * 50)
    run_with_timeout(parent_dir, threshold_bytes, timeout, mode)

if __name__ == "__main__":
    main()
