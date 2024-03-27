from pathlib import Path; print("loc: ", sum([len(open(f).readlines()) for f in Path("tinyexplain").glob("**/*.py")]))
