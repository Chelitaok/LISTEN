"""Quick check that Whisper and spaCy are installed and working."""
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print(f"\nPython {sys.version}\n")

ok = True
ok &= check("torch", lambda: __import__("torch").__version__)
ok &= check("torch MPS available", lambda: str(__import__("torch").backends.mps.is_available()))
ok &= check("whisper", lambda: __import__("whisper").__version__)
ok &= check("whisperx", lambda: (__import__("whisperx") and "OK"))
ok &= check("pyannote.audio", lambda: __import__("pyannote.audio", fromlist=["__version__"]).__version__)
ok &= check("spacy", lambda: __import__("spacy").__version__)
ok &= check("spacy model en_core_web_sm", lambda: str(__import__("spacy").load("en_core_web_sm").meta["version"]))
ok &= check("ffmpeg", lambda: __import__("subprocess").check_output(["ffmpeg", "-version"]).split()[2].decode())

print()
if ok:
    print("All checks passed. Environment is ready.")
else:
    print("Some checks failed — see above.")
    sys.exit(1)
