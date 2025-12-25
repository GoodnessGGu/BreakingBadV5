import os
from dotenv import load_dotenv

print("--- Debug Env ---")
print(f"Current Directory: {os.getcwd()}")
files = os.listdir('.')
print(f"Files: {files}")

if '.env' in files:
    print(".env file found!")
    with open('.env', 'r') as f:
        print("--- Content Preview (Keys only) ---")
        for line in f:
            if '=' in line:
                key = line.split('=')[0].strip()
                print(f"Found Key: '{key}'")
else:
    print(".env file NOT found!")

print("--- Loading Dotenv ---")
load_dotenv()
print(f"os.getenv('email'): {os.getenv('email')}")
print(f"os.getenv('password'): {os.getenv('password')}")
print(f"os.getenv('EMAIL'): {os.getenv('EMAIL')}")
print(f"os.getenv('PASSWORD'): {os.getenv('PASSWORD')}")
