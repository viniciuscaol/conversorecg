from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import os

app = Flask(__name__)

# --- O código de geração de ECG que já desenvolvemos ---
# Esta função será chamada pelo endpoint da API
def gerar_ecg_do_xml_interno(xml_content):
    try:
        root = ET.parse(io.StringIO(xml_content)).getroot()

        registros_tag = root.find(".//Registros")
        taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
        sensibilidade_str = registros_tag.get("Sensibilidade")

        taxa_amostragem = float(re.search(r'\d+', taxa_amostragem_str).group())
        sensibilidade_valor = 0.005 # mV por unidade de amostra

        ecg_data = {}
        for canal in root.findall(".//Canal"):
            nome = canal.get("Nome")
            amostras_str = canal.find("Amostras")
            if amostras_str is None or amostras_str.text is None:
                continue
            
            raw_samples_text = amostras_str.text.replace('\r', '').replace('\n', '')
            raw_samples = np.array([float(x) for x in raw_samples_text.split(';') if x.strip()])
            samples_mv = raw_samples * sensibilidade_valor
            ecg_data[nome] = samples_mv

        if not ecg_data:
            raise ValueError("No valid ECG data found after parsing channels.")

        paciente_nome = root.find(".//Paciente/Nome")
        exam_data = root.find(".//Exame/Data")
        exam_hora = root.find(".//Exame/Hora")
        paciente_data_nascimento = root.find(".//Paciente/DataNascimento")
        paciente_sexo = root.find(".//Paciente/Sexo")

        nome_paciente = paciente_nome.text if paciente_nome is not None else "Desconhecido"
        data_exame = exam_data.text if exam_data is not None else "N/A"
        hora_exame = exam_hora.text if exam_hora is not None else "N/A"
        sexo_paciente = paciente_sexo.text if paciente_sexo is not None else "N/A"
        
        idade_paciente = "N/A"
        if paciente_data_nascimento is not None and paciente_data_nascimento.text:
            try:
                ano_nascimento = int(paciente_data_nascimento.text.split('/')[-1])
                ano_exame = int(data_exame.split('/')[-1])
                idade_paciente = ano_exame - ano_nascimento
            except (ValueError, IndexError):
                idade_paciente = "N/A"

        num_amostras = len(list(ecg_data.values())[0])
        eixo_tempo = np.arange(num_amostras) / taxa_amostragem

        num_derivações = len(ecg_data)
        fig, axes = plt.subplots(num_derivações, 1, figsize=(15, 2.5 * num_derivações), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f"ECG - Paciente: {nome_paciente} ({sexo_paciente}, {idade_paciente} anos) - Data: {data_exame} {hora_exame}", fontsize=16)

        all_samples = np.concatenate(list(ecg_data.values()))
        ymin_global = np.floor(all_samples.min() / 0.5) * 0.5
        ymax_global = np.ceil(all_samples.max() / 0.5) * 0.5
        if abs(ymax_global - ymin_global) < 1.0:
            mid_point = (ymax_global + ymin_global) / 2
            ymin_global = mid_point - 0.5
            ymax_global = mid_point + 0.5

        for i, (nome_derivação, amostras_mv) in enumerate(ecg_data.items()):
            ax = axes[i] if num_derivações > 1 else axes
            ax.plot(eixo_tempo, amostras_mv, linewidth=0.7, color='black')
            ax.set_ylabel(f"{nome_derivação} (mV)")

            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.2))
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.01, 0.5))
            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.04), minor=True)
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.001, 0.1), minor=True)

            ax.grid(True, which='major', color='red', linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', color='pink', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='both', labelsize=8)
            ax.set_ylim(ymin_global, ymax_global)

        if num_derivações > 1:
            axes[-1].set_xlabel("Tempo (s)")
        else:
            axes.set_xlabel("Tempo (s)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salva o gráfico em um buffer de memória em vez de um arquivo no disco
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0) # Volta ao início do buffer
        plt.close(fig) # Fecha a figura para liberar memória
        return buf

    except Exception as e:
        print(f"Erro na geração do ECG: {e}")
        return None

# --- Rotas do Flask ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    if 'ecg_file' not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files['ecg_file']
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400
    
    if file:
        xml_content = file.read().decode('utf-8') # Lê o conteúdo do arquivo como string UTF-8
        
        image_buffer = gerar_ecg_do_xml_interno(xml_content)
        
        if image_buffer:
            return send_file(image_buffer, mimetype='image/png', as_attachment=False)
        else:
            return "Erro ao gerar o gráfico de ECG. Verifique o formato do XML.", 500

if __name__ == '__main__':
    # Para rodar localmente, certifique-se de ter Flask instalado: pip install Flask
    # E rodar o arquivo: python app.py
    # O servidor estará disponível em http://127.0.0.1:5000/
    app.run(debug=True) # debug=True recarrega o servidor automaticamente em mudanças