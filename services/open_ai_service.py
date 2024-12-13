import os
from openai import AzureOpenAI

def get_openai_response(message):
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint="https://tohuu-m4jpteo9-swedencentral.openai.azure.com/",
        api_key="Ex0jctDBC7HmWTmAsKy0oBbez1TevQLTwsFfCd7FxnUmKOFZGL0SJQQJ99ALACfhMk5XJ3w3AAAAACOGBlbf",
        api_version="2024-07-18"
    )
    
    print("4iYvyD9x7Nof12sWfslnIcebZbnHo3NO6kkUgQF1Sj0XXoaIvtdcJQQJ99ALACHYHv6XJ3w3AAAAACOGM7tJ")
    
    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        
        # Extract and return the response content
        result = response.choices[0].message.content
        print(result)
    
    except Exception as e:
        print(f"Error while communicating with OpenAI: {str(e)}")
        return None
    
get_openai_response("Hello")
