from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from rag_manager import add_files_to_rag

class CodeWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        print(f"File modified: {file_path}")
        add_files_to_rag([file_path])

def start_watcher(path):
    event_handler = CodeWatcher()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
