import cv2
import numpy as np
import os

class ProcessadorImagem:

    def _salvar_diagnosticos(self, gray, blurred, thresh, thresh_filled, opening, sure_bg, dist_transform, sure_fg, unknown):
        pasta_diag = "resultados/diagnostico"
        os.makedirs(pasta_diag, exist_ok=True)
        cv2.imwrite(f"{pasta_diag}/1_gray.jpg", gray)
        cv2.imwrite(f"{pasta_diag}/2_blurred.jpg", blurred)
        cv2.imwrite(f"{pasta_diag}/3_thresh.jpg", thresh)
        cv2.imwrite(f"{pasta_diag}/4_thresh_filled.jpg", thresh_filled)
        cv2.imwrite(f"{pasta_diag}/5_opening.jpg", opening)
        cv2.imwrite(f"{pasta_diag}/6_sure_bg.jpg", sure_bg)
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"{pasta_diag}/7_dist_transform.jpg", dist_norm)
        cv2.imwrite(f"{pasta_diag}/8_sure_fg.jpg", sure_fg)
        cv2.imwrite(f"{pasta_diag}/9_unknown.jpg", unknown)

    def detectar_fator_escala(self, img, tamanho_real_cm=1.0):
        """
        Detecta o quadrado rosa e retorna (px_por_cm, bbox_do_quadrado).
        bbox é (x, y, w, h) usado para excluir o quadrado dos contornos de sementes.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Rosa/magenta: cobre duas faixas do HSV (abaixo de 10 e acima de 140)
        mascara1 = cv2.inRange(hsv, np.array([140, 50,  80]), np.array([180, 255, 255]))
        mascara2 = cv2.inRange(hsv, np.array([  0, 50,  80]), np.array([ 10, 255, 255]))
        mascara_rosa = cv2.bitwise_or(mascara1, mascara2)

        kernel = np.ones((5, 5), np.uint8)
        mascara_rosa = cv2.morphologyEx(mascara_rosa, cv2.MORPH_CLOSE, kernel)
        mascara_rosa = cv2.morphologyEx(mascara_rosa, cv2.MORPH_OPEN,  kernel)

        # Salva para diagnóstico — útil para calibrar a faixa HSV
        os.makedirs("resultados/diagnostico", exist_ok=True)
        cv2.imwrite("resultados/diagnostico/0_mascara_rosa.jpg", mascara_rosa)

        contornos, _ = cv2.findContours(mascara_rosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            print("[Escala] Quadrado rosa não encontrado.")
            return None, None

        c = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < 500:
            print(f"[Escala] Contorno rosa muito pequeno ({area:.0f}px²), ignorado.")
            return None, None

        # Verifica se é razoavelmente quadrado (razão de aspecto próxima de 1)
        retangulo = cv2.minAreaRect(c)
        _, (w, h), _ = retangulo
        if w == 0 or h == 0:
            return None, None

        razao = max(w, h) / min(w, h)
        if razao > 1.5:
            print(f"[Escala] Contorno rosa não é quadrado (razão {razao:.2f}), ignorado.")
            return None, None

        lado_px = (w + h) / 2
        px_por_cm = lado_px / tamanho_real_cm

        # bbox para exclusão posterior
        x, y, bw, bh = cv2.boundingRect(c)
        bbox_rosa = (x, y, bw, bh)

        print(f"[Escala] Quadrado detectado: {lado_px:.1f}px → {px_por_cm:.2f} px/cm | bbox={bbox_rosa}")
        return px_por_cm, bbox_rosa

    def _contorno_dentro_bbox(self, contorno, bbox, margem=10):
        """Retorna True se o centro do contorno estiver dentro da bbox do quadrado rosa."""
        if bbox is None:
            return False
        x, y, bw, bh = bbox
        M = cv2.moments(contorno)
        if M["m00"] == 0:
            return False
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (x - margem) < cx < (x + bw + margem) and (y - margem) < cy < (y + bh + margem)

    def obter_contornos(self, caminho_imagem):
        img = cv2.imread(caminho_imagem)
        if img is None:
            raise FileNotFoundError(f"Erro: Imagem não encontrada em {caminho_imagem}")

        # 1. Detecta escala ANTES do watershed
        px_por_cm, bbox_rosa = self.detectar_fator_escala(img, tamanho_real_cm=1.0)

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)

        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        thresh_filled = thresh | cv2.bitwise_not(im_floodfill)

        kernel  = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh_filled, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg        = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg     = cv2.threshold(dist_transform, 0.50 * dist_transform.max(), 255, 0)
        sure_fg        = np.uint8(sure_fg)
        unknown        = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)

        contornos_separados = []
        for marker_id in np.unique(markers):
            if marker_id <= 1:
                continue
            mascara_individual = np.zeros(gray.shape, dtype="uint8")
            mascara_individual[markers == marker_id] = 255
            cnts, _ = cv2.findContours(mascara_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 150:
                    # 2. Exclui o contorno que corresponde ao quadrado rosa
                    if self._contorno_dentro_bbox(c, bbox_rosa):
                        print(f"[Escala] Contorno excluído (é o quadrado de referência)")
                        continue
                    contornos_separados.append(c)

        self._salvar_diagnosticos(gray, blurred, thresh, thresh_filled, opening, sure_bg, dist_transform, sure_fg, unknown)

        return img, gray, opening, contornos_separados, px_por_cm

    """def obter_contornos(self, caminho_imagem):
        img = cv2.imread(caminho_imagem)
        if img is None:
            raise FileNotFoundError(f"Erro: Imagem não encontrada em {caminho_imagem}")

        px_por_cm, bbox_rosa = self.detectar_fator_escala(img, tamanho_real_cm=1.0)

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # kernel maior — suaviza mais

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        thresh_filled = thresh | cv2.bitwise_not(im_floodfill)

        kernel  = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh_filled, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg        = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # ── Ponto crítico: threshold mais baixo separa sementes próximas ──
        # 0.5 é agressivo demais quando sementes se tocam levemente
        # 0.3 preserva mais regiões individuais
        _, sure_fg = cv2.threshold(dist_transform, 0.30 * dist_transform.max(), 255, 0)
        sure_fg    = np.uint8(sure_fg)
        unknown    = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)

        # ── Limiar de área menor — captura sementes pequenas/parciais ──
        # O valor anterior de 150 descartava sementes menores da borda
        LIMIAR_AREA = 80

        contornos_separados = []
        for marker_id in np.unique(markers):
            if marker_id <= 1:
                continue
            mascara_individual = np.zeros(gray.shape, dtype="uint8")
            mascara_individual[markers == marker_id] = 255
            cnts, _ = cv2.findContours(mascara_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > LIMIAR_AREA:
                    if self._contorno_dentro_bbox(c, bbox_rosa):
                        print(f"[Escala] Contorno excluído (quadrado de referência)")
                        continue
                    contornos_separados.append(c)

        self._salvar_diagnosticos(gray, blurred, thresh, thresh_filled, opening,
                                sure_bg, dist_transform, sure_fg, unknown)

        return img, gray, opening, contornos_separados, px_por_cm"""