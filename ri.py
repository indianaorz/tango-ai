import unicodedata
import pyperclip # You might need to install this library: pip install pyperclip

def remove_invisible_chars_preserve_newlines(text):
    """
    Removes invisible Unicode characters from a string,
    while preserving newline characters.
    """
    cleaned_text = []
    for char in text:
        # Explicitly keep newline characters
        if char in ('\n', '\r'):
            cleaned_text.append(char)
            continue

        # Categories of characters often considered "invisible" or problematic
        # Zs: Space separator (includes non-breaking space, but we often want to keep regular space)
        # Cc: Other, control (excluding CR and LF which we handle above)
        # Cf: Other, format (includes zero-width characters)
        # Mn: Mark, nonspacing
        # Mc: Mark, spacing combining
        # Me: Mark, enclosing
        category = unicodedata.category(char)

        if not (
            category in ('Cc', 'Cf', 'Mn', 'Mc', 'Me') or
            # Specifically target zero-width spaces and similar format characters
            # U+200B: Zero Width Space
            # U+200C: Zero Width Non-Joiner
            # U+200D: Zero Width Joiner
            # U+FEFF: Zero Width No-Break Space (BOM)
            # Add more specific problematic unicode points if needed
            ord(char) in (0x200B, 0x200C, 0x200D, 0xFEFF)
        ):
            cleaned_text.append(char)

    return "".join(cleaned_text)

try:
    # 1. Get text from clipboard
    original_text = pyperclip.paste()

    if original_text:
        # 2. Remove invisible characters, preserving newlines
        cleaned_text = remove_invisible_chars_preserve_newlines(original_text)

        if original_text != cleaned_text:
            # 3. Copy cleaned text back to clipboard
            pyperclip.copy(cleaned_text)
            print("Invisible characters removed (newlines preserved). Clipboard updated.")
        else:
            print("No invisible characters (to remove) found or no changes made.")
    else:
        print("Clipboard is empty.")

except pyperclip.PyperclipException as e:
    print(f"Error with clipboard operations: {e}")
    print("Please make sure you have a copy/paste mechanism installed.")
    print("On Linux, you can try: sudo apt-get install xclip or sudo apt-get install xsel")
    print("On Windows and macOS, pyperclip usually works out of the box.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")