# Sistema de Fenotipagem Web

## O que mudou em relação ao pré-TCC

| Agora                              |
|------------------------------------|
| Computador comum + celular         |
| Câmera do celular via browser      |
| **api.py** (Flask)                 |
| Acessar site no celular            |

---

## Estrutura de arquivos

```
projeto/
├── api.py                      
├── processamento_imagem.py    
├── extracao_dados.py           
├── exportador_pdf.py           
├── main.py                    
├── requirements.txt           
├── templates/
│   └── index.html             
└── resultados/                 
```

---

## Como instalar

```bash
pip install -r requirements.txt
```

---

## Como rodar

### 1. Descubra o IP do seu computador

**Windows:**
```
ipconfig
```
Procure por "Endereço IPv4" (ex: `192.168.1.15`)

**Linux/Mac:**
```
ip addr   ou   ifconfig
```

### 2. Inicie a API

```bash
python api.py
```

Você verá algo como:
```
* Running on http://0.0.0.0:5000
```

### 3. Acesse pelo celular

O celular e o computador precisam estar **no mesmo Wi-Fi**.

Abra o navegador do celular e acesse:
```
http://192.168.1.15:5000
```
(use o IP do seu computador)

> A página `index.html` também pode ser aberta diretamente no browser
> do celular sem servidor, mas nesse caso ela não pode enviar imagens
> para a API — é melhor servir pelo Flask mesmo.

---

## Adicionando a rota para servir o HTML (opcional)

Se quiser que a API já sirva o HTML automaticamente, adicione isso no `api.py`:

```python
from flask import render_template

@app.route("/")
def index():
    return render_template("index.html")
```

Aí você acessa `http://IP:5000/` e já cai direto na interface.

---

## Fluxo resumido

```
Celular (browser)
    │ foto via <input capture="environment">
    │ POST /analisar  (multipart/form-data)
    ▼
api.py (Flask)
    │ decodifica imagem com OpenCV
    │ chama processamento_imagem.py
    │ chama extracao_dados.py
    │ chama exportador_pdf.py
    │ retorna JSON com dados + imagem marcada (base64)
    ▼
Celular (browser)
    │ exibe tabela de sementes
    │ exibe imagem com contornos marcados
    └ botão para baixar PDF  →  GET /relatorio
```

---

## Testando sem celular (com Postman ou curl)

```bash
curl -X POST http://localhost:5000/analisar \
  -F "imagem=@sua_foto.jpg"
```
