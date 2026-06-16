from openai import OpenAI

cliente = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

resposta = cliente.chat.completions.create(
    model="llama3.1:8b",
   #model=" llama3.2-vision:11b",

    messages=[
        {"role": "user", "content": "Qual é a capital da França?"}
    ],
    temperature=0.7
)

print(resposta.choices[0].message.content)
