import os
import json
import random
import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Set your OpenAI API key here
# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
N_RETRIES = 3

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt_description, prev_examples, temperature=0.5):
    """
    Generates one prompt/response example for fine-tuning.
    It optionally includes a few previous examples to encourage diversity.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are generating data which will be used to train a machine learning model.\n\n"
                "You will be given a high-level description of the model we want to train, and from that, "
                "you will generate data samples, each with a prompt/response pair.\n\n"
                "You will do so in this format:\n\n"
                "prompt\n-----------\n$prompt_goes_here\n-----------\n\n"
                "response\n-----------\n$response_goes_here\n-----------\n\n"
                "Only one prompt/response pair should be generated per turn.\n\n"
                "For each turn, make the example slightly more complex than the last, while ensuring diversity.\n\n"
                "Make sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\n"
                f"Here is the type of model we want to train:\n{prompt_description}"
            )
        }
    ]
    
    # Optionally include a few previous examples to add diversity
    if prev_examples:
        # Limit to 8 previous examples if there are many
        examples_to_include = random.sample(prev_examples, min(8, len(prev_examples)))
        for ex in examples_to_include:
            messages.append({
                "role": "assistant",
                "content": ex
            })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=750,
    )
    return response.choices[0].message['content']

def generate_system_message(prompt_description, temperature=0.5):
    """
    Generates a concise system message that the fine-tuned model will use during inference.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. "
                    "Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. "
                    "A good format to follow is: Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO..\n\n"
                    "Make it as concise as possible. Include nothing but the system prompt in your response.\n\n"
                    "For example, never write: \"$SYSTEM_PROMPT_HERE\".\n\n"
                    "It should be like: $SYSTEM_PROMPT_HERE."
                )
            },
            {
                "role": "user",
                "content": prompt_description.strip(),
            }
        ],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message['content']

def generate_examples(prompt_description, num_examples, temperature=0.5):
    """
    Generates a list of training examples using the generate_example function.
    """
    prev_examples = []
    for i in range(num_examples):
        print(f'Generating example {i + 1} of {num_examples}...')
        example = generate_example(prompt_description, prev_examples, temperature)
        prev_examples.append(example)
    return prev_examples

def parse_examples_to_dataframe(examples):
    """
    Parses the generated examples to extract prompt and response parts,
    then stores them in a pandas DataFrame.
    """
    prompts = []
    responses = []
    
    for example in examples:
        try:
            # Split by the defined delimiter
            split_example = example.split('-----------')
            # Expected format:
            # split_example[0] -> "prompt" (or empty)
            # split_example[1] -> actual prompt
            # split_example[2] -> (possibly a header for response)
            # split_example[3] -> actual response
            prompt_text = split_example[1].strip()
            response_text = split_example[3].strip()
            prompts.append(prompt_text)
            responses.append(response_text)
        except Exception as e:
            print("Failed to parse example:", example)
            continue

    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })
    df.drop_duplicates(inplace=True)
    print(f'There are {len(df)} successfully-generated examples.')
    return df

def create_training_examples(df, system_message):
    """
    Formats the DataFrame rows into training examples for fine-tuning.
    Each example contains the system message, the prompt (user message),
    and the response (assistant message).
    """
    training_examples = []
    for _, row in df.iterrows():
        training_example = {
            "messages": [
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['response']}
            ]
        }
        training_examples.append(training_example)
    return training_examples

def save_training_examples(training_examples, filename='training_examples.jsonl'):
    """
    Saves the training examples to a JSONL file, which is the format required for OpenAI fine-tuning.
    """
    with open(filename, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    print(f"Training examples saved to {filename}")

def upload_file_for_fine_tune(filename):
    """
    Uploads the training examples file to OpenAI for fine-tuning.
    """
    file_response = openai.File.create(
      file=open(filename, "rb"),
      purpose='fine-tune'
    )
    file_id = file_response.id
    print("File uploaded. File ID:", file_id)
    return file_id

def main():
    # Get user input for the model description, temperature, and number of examples
    prompt_description = input("Enter the high-level description for the model: ").strip()
    temperature_input = input("Enter temperature (e.g., 0.5) [default 0.5]: ").strip()
    num_examples_input = input("Enter number of examples to generate [default 10]: ").strip()
    
    temperature = float(temperature_input) if temperature_input else 0.5
    num_examples = int(num_examples_input) if num_examples_input else 10

    # Generate a system message for inference
    print("Generating system message...")
    system_message = generate_system_message(prompt_description, temperature)
    print("System message generated:")
    print(system_message)
    
    # Generate training examples based on the model description
    print("Generating training examples...")
    examples = generate_examples(prompt_description, num_examples, temperature)
    
    # Parse the examples into a DataFrame to extract prompt/response pairs
    print("Parsing examples...")
    df = parse_examples_to_dataframe(examples)
    
    # Create the training examples in the required format for fine-tuning
    print("Creating training examples...")
    training_examples = create_training_examples(df, system_message)
    
    # Save the training examples to a JSONL file
    filename = 'training_examples.jsonl'
    save_training_examples(training_examples, filename)
    
    # Upload the JSONL file to OpenAI for fine-tuning
    print("Uploading training examples file for fine-tuning...")
    upload_file_for_fine_tune(filename)
    
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
