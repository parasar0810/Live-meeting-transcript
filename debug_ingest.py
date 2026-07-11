import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", """
import sys, traceback
try:
    import chromadb
    print("chromadb imported OK")
    c = chromadb.PersistentClient(path='knowledge_base/chroma_db')
    print("client created OK")
    col = c.get_or_create_collection('meeting_kb')
    print(f"collection count: {col.count()}")
except:
    traceback.print_exc()
"""],
    capture_output=True,
    text=True,
    cwd=r"c:\Users\KIIT0001\Desktop\live meeting project"
)
with open(r"c:\Users\KIIT0001\Desktop\live meeting project\debug_output4.txt", "w") as f:
    f.write(f"RETURNCODE: {result.returncode}\n")
    f.write(f"STDOUT:\n{result.stdout}\n")
    f.write(f"STDERR:\n{result.stderr}\n")
print("Done - check debug_output4.txt")
