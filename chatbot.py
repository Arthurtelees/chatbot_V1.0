# Chatbot com Transformers e Banco de Dados
# Autor: Arthur Teles
# Descrição: Chatbot em português com memória de conversa e aprendizagem contínua

from transformers import pipeline, AutoTokenizer
import re
from collections import deque
import sqlite3
from datetime import datetime

class ChatbotDatabase:
    def __init__(self, db_name='chatbot_memory.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.commit()

    def save_conversation(self, user_input, bot_response):
        self.cursor.execute('''
        INSERT INTO conversations (user_input, bot_response)
        VALUES (?, ?)''', (user_input, bot_response))
        self.conn.commit()

    def close(self):
        self.conn.close()

class ChatbotPortugues:
    def __init__(self):
        self.model_name = "pierreguillou/gpt2-small-portuguese" 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.chatbot = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device='cpu'
        )
        self.historico = deque(maxlen=4)
        self.config_resposta = {
            'max_length': 120,
            'temperature': 0.85,
            'repetition_penalty': 1.3,
            'top_p': 0.95
        }
        self.db = ChatbotDatabase()

    def limpar_resposta(self, texto):
        return texto.split('\n')[0].strip()

    def responder(self, pergunta):
        if pergunta.lower() in ['sair', 'exit', 'bye']:
            self.db.close()
            return "Até logo! Foi bom conversar com você."
            
        self.historico.append(f"Você: {pergunta}")
        contexto = "\n".join(self.historico) + "\nBot:"
        
        resposta = self.chatbot(
            contexto,
            **self.config_resposta
        )[0]['generated_text']
        
        resposta_limpa = self.limpar_resposta(resposta.split("Bot:")[-1])
        
        self.db.save_conversation(pergunta, resposta_limpa)
        
        self.historico.append(f"Bot: {resposta_limpa}")
        return resposta_limpa

if __name__ == "__main__":
    bot = ChatbotPortugues()
    print("Chatbot Português (Digite 'sair' para encerrar)\n")
    
    while True:
        try:
            mensagem = input("Você: ")
            resposta = bot.responder(mensagem)
            print(f"Bot: {resposta}\n")
            
            if mensagem.lower() in ['sair', 'exit']:
                break
                
        except KeyboardInterrupt:
            print("\nEncerrando o chatbot...")
            bot.db.close()
            break