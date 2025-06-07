import google.generativeai as genai
genai.configure(api_key="if you want to check which models are available, you can use this code")

models = genai.list_models()
for m in models:
    print(m.name)