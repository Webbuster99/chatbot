# import os
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# load_dotenv()

# # Set your API key (better to use environment variables for security)
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# try:
#     # Generate content with both text and image responses
#     response = client.models.generate_content(
#         model="gemini-2.0-flash-exp",
#         contents=(
#             "Generate a story about a cute baby turtle in a 3d digital art style. "
#             "For each scene, generate an image."
#         ),
#         config=types.GenerateContentConfig(
#             response_modalities=["Text", "Image"]
#         ),
#     )
    
#     # Print the response
#     print("Response received:")
    
#     # Inspect the response structure
#     print(f"Response type: {type(response)}")
#     print(f"Response attributes: {dir(response)}")
    
#     # Try different ways to access the content
#     if hasattr(response, 'text'):
#         print(f"TEXT: {response.text}")
    
#     if hasattr(response, 'candidates'):
#         for i, candidate in enumerate(response.candidates):
#             print(f"Candidate {i}:")
#             if hasattr(candidate, 'content'):
#                 content = candidate.content
#                 print(f"Content type: {type(content)}")
#                 print(f"Content attributes: {dir(content)}")
                
#                 # Try to access parts if available
#                 if hasattr(content, 'parts'):
#                     for j, part in enumerate(content.parts):
#                         print(f"Part {j} type: {type(part)}")
#                         print(f"Part {j} attributes: {dir(part)}")
                        
#                         # Check for text
#                         if hasattr(part, 'text') and part.text:
#                             print(f"TEXT: {part.text}")
                        
#                         # Check for image
#                         if hasattr(part, 'inline_data') and part.inline_data:
#                             print(f"IMAGE: Found image data")
#                             # Optionally save the image
#                             # img_data = base64.b64decode(part.inline_data.data)
#                             # img = Image.open(io.BytesIO(img_data))
#                             # img.save(f"turtle_scene_{i}_{j}.jpg")

# except Exception as e:
#     print(f"Error occurred: {e}")