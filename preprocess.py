import re
import json
import os
import argparse

def txt_to_json(txt_file_path, json_file_path, whatsapp_name):
    """
    Convert WhatsApp chat text file to JSON format.
    """
    # Initialize the conversations list
    conversations = []

    # Regular expression pattern to match chat lines
    # Matches both cases with and without seconds in the time format
    pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}(:\d{2})?)\] (.+?): (.+)"

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        file.readline()  # Skip the first line
        for line in file:
            match = re.match(pattern, line)
            if match:
                # Extract relevant components
                sender = match.group(4).strip()
                content = match.group(5).strip()

                # Filter out "Missed call" messages
                if "Missed video call" in content or "Missed voice call" in content:
                    continue

                # Determine if the message is from the user or someone else
                from_field = "gpt" if whatsapp_name in sender else "human"

                conversations.append({
                    "from": from_field,
                    "value": content
                })

    # Save conversations to JSON, splitting into chunks if too long
    if len(conversations) > 40:
        for i in range(0, len(conversations), 40):
            partial_conversations = conversations[i:i+40]
            if len(partial_conversations) < 3:
                continue
            json.dump({"conversations": partial_conversations},
                      open(f"{json_file_path}_{i}.json", 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=4)
    else:
        if len(conversations) < 3:
            return
        # Ensure the file has a `.json` extension
        if not json_file_path.endswith('.json'):
            json_file_path += '.json'
        json.dump({"conversations": conversations},
                  open(json_file_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)


def process_folder(data_folder, output_folder, whatsapp_name):
    """
    Process all .txt files in the data folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(data_folder, filename)
            json_file_name = os.path.splitext(filename)[0] + ".json"  # Ensure .json extension
            json_file_path = os.path.join(output_folder, json_file_name)
            txt_to_json(txt_file_path, json_file_path, whatsapp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WhatsApp chat .txt files to JSON format.")
    parser.add_argument("whatsapp_name", type=str, help="Your WhatsApp name to identify messages from you")
    args = parser.parse_args()

    if not args.whatsapp_name:
        print("Please provide your WhatsApp name.")
        exit()

    data_folder = "data/raw_data"
    output_folder = "data/preprocessed"
    process_folder(data_folder, output_folder, args.whatsapp_name)
