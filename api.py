"""
api.py  -  Backend Flask do sistema de fenotipagem
Recebe a foto do celular via HTTP, executa o pipeline
e devolve JSON com:
- total de sementes
- tabela de sementes
- imagens das etapas do processamento
- link do PDF
- recortes das sementes isoladas
"""

import os
import time
import base64
import zipfile
import io

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS

from processamento_imagem import ProcessadorImagem
from extracao_dados import ExtratorCaracteristicas
from exportador_pdf import ExportadorRelatorio

app = Flask(__name__, template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
DIAG_DIR = os.path.join(RESULTADOS_DIR, "diagnostico")
SEMENTES_DIR = os.path.join(RESULTADOS_DIR, "sementes_isoladas")

os.makedirs(SEMENTES_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)


def caminho_para_url(caminho_relativo: str, version: int | None = None) -> str:
    caminho_relativo = caminho_relativo.replace("\\", "/")
    url = url_for("servir_arquivo", caminho=caminho_relativo, _external=True)
    if version is not None:
        url = f"{url}?v={version}"
    return url


@app.route("/", methods=["GET"])
def index():
    caminho = os.path.join(BASE_DIR, "templates", "index.html")
    return send_file(caminho)


@app.route("/arquivo/<path:caminho>", methods=["GET"])
def servir_arquivo(caminho):
    caminho_absoluto = os.path.abspath(os.path.join(BASE_DIR, caminho))

    if not caminho_absoluto.startswith(BASE_DIR):
        return jsonify({"erro": "Caminho invalido."}), 400

    if not os.path.exists(caminho_absoluto):
        return jsonify({"erro": "Arquivo nao encontrado."}), 404

    return send_file(caminho_absoluto)


@app.route("/analisar", methods=["POST"])
def analisar():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem recebida. Envie o campo 'imagem'."}), 400

    arquivo = request.files["imagem"]
    if arquivo.filename == "":
        return jsonify({"erro": "Nome de arquivo vazio."}), 400

    bytes_imagem = arquivo.read()
    arr = np.frombuffer(bytes_imagem, np.uint8)
    img_original = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_original is None:
        return jsonify({"erro": "Nao foi possivel decodificar a imagem. Verifique o formato."}), 422

    caminho_temp = os.path.join(RESULTADOS_DIR, "captura_atual.jpg")
    cv2.imwrite(caminho_temp, img_original)

    processador = ProcessadorImagem()
    extrator = ExtratorCaracteristicas()
    exportador = ExportadorRelatorio()

    try:
        img_original, img_cinza, img_binarizada, contornos, px_por_cm = processador.obter_contornos(caminho_temp)
    except Exception as e:
        return jsonify({"erro": f"Falha no processamento da imagem: {str(e)}"}), 500

    cv2.imwrite(os.path.join(RESULTADOS_DIR, "1_imagem_original.jpg"), img_original)
    cv2.imwrite(os.path.join(RESULTADOS_DIR, "2_imagem_tons_cinza.jpg"), img_cinza)
    cv2.imwrite(os.path.join(RESULTADOS_DIR, "3_imagem_opening.jpg"), img_binarizada)

    lista_sementes = extrator.extrair_dados(img_original, contornos, px_por_cm=px_por_cm)

    img_marcacoes = img_original.copy()
    for s in lista_sementes:
        cv2.drawContours(img_marcacoes, [s.contorno], -1, (0, 255, 0), 2)
        cv2.putText(
            img_marcacoes,
            f"ID:{s.id_semente}",
            (s.centro_x - 15, s.centro_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    cv2.imwrite(os.path.join(RESULTADOS_DIR, "3_imagem_marcacoes.jpg"), img_marcacoes)

    _, buffer = cv2.imencode(".jpg", img_marcacoes)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    exportador.gerar_pdf(lista_sementes)

    timestamp = int(time.time())
    unidade_atual = "mm" if px_por_cm else "px"

    sementes_json = []
    for s in lista_sementes:
        sementes_json.append({
            "id": s.id_semente,
            "area": round(s.area, 2),
            "perimetro": round(s.perimetro, 2),
            "comprimento": round(s.comprimento, 2),
            "largura": round(s.largura, 2),
            "circularidade": round(s.circularidade, 4),
            "razao_aspecto": round(s.razao_aspecto, 4),
            "cor_rgb": s.cor_rgb,
            "unidade": s.unidade,
            "centro_x": s.centro_x,
            "centro_y": s.centro_y,
            "recorte_url": caminho_para_url(s.caminho_recorte, timestamp) if s.caminho_recorte else None,
        })

    imagens = {
        "original": caminho_para_url("resultados/1_imagem_original.jpg", timestamp),
        "cinza": caminho_para_url("resultados/diagnostico/1_gray.jpg", timestamp),
        "blurred": caminho_para_url("resultados/diagnostico/2_blurred.jpg", timestamp),
        "thresh": caminho_para_url("resultados/diagnostico/3_thresh.jpg", timestamp),
        "thresh_filled": caminho_para_url("resultados/diagnostico/4_thresh_filled.jpg", timestamp),
        "opening": caminho_para_url("resultados/diagnostico/5_opening.jpg", timestamp),
        "sure_bg": caminho_para_url("resultados/diagnostico/6_sure_bg.jpg", timestamp),
        "dist_transform": caminho_para_url("resultados/diagnostico/7_dist_transform.jpg", timestamp),
        "sure_fg": caminho_para_url("resultados/diagnostico/8_sure_fg.jpg", timestamp),
        "unknown": caminho_para_url("resultados/diagnostico/9_unknown.jpg", timestamp),
        "marcacoes": caminho_para_url("resultados/3_imagem_marcacoes.jpg", timestamp),
    }

    return jsonify({
        "total_sementes": len(lista_sementes),
        "escala": {                                       
            "px_por_cm": round(px_por_cm, 2) if px_por_cm else None,
            "unidade": unidade_atual
        },
        "sementes": sementes_json,
        "imagem_marcacoes_base64": img_base64,
        "imagens": imagens,
        "pdf_url": caminho_para_url("resultados/relatorio_fenotipagem.pdf", timestamp),
    })


@app.route("/relatorio", methods=["GET"])
def baixar_relatorio():
    caminho_pdf = os.path.join(RESULTADOS_DIR, "relatorio_fenotipagem.pdf")
    if not os.path.exists(caminho_pdf):
        return jsonify({"erro": "Nenhum relatorio disponivel. Faca uma analise primeiro."}), 404
    return send_file(caminho_pdf, as_attachment=True, download_name="relatorio_fenotipagem.pdf")


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "ok", "mensagem": "API de fenotipagem ativa."})

@app.route("/download/sementes", methods=["GET"])
def download_sementes():
    pasta = SEMENTES_DIR
    if not os.path.exists(pasta) or not os.listdir(pasta):
        return jsonify({"erro": "Nenhuma semente disponível."}), 404

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for arquivo in sorted(os.listdir(pasta)):
            if arquivo.endswith(".png"):
                zf.write(os.path.join(pasta, arquivo), arquivo)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="sementes_isoladas.zip",
                     mimetype="application/zip")


@app.route("/download/diagnostico", methods=["GET"])
def download_diagnostico():
    pasta = DIAG_DIR
    if not os.path.exists(pasta) or not os.listdir(pasta):
        return jsonify({"erro": "Nenhum diagnóstico disponível."}), 404

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for arquivo in sorted(os.listdir(pasta)):
            if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
                zf.write(os.path.join(pasta, arquivo), arquivo)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="diagnostico.zip",
                     mimetype="application/zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
