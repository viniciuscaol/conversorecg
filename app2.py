from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import os

app = Flask(__name__)

# --- O código de geração de ECG que já desenvolvemos, adaptado ---
def gerar_ecg_do_xml_interno(xml_content):
    try:
        root = ET.parse(io.StringIO(xml_content)).getroot()
        print("XML parsed successfully.")

        registros_tag = root.find(".//Registros")
        if registros_tag is None:
            raise ValueError("Tag 'Registros' not found in XML.")

        taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
        sensibilidade_str = registros_tag.get("Sensibilidade")
        velocidade_exame = root.find(".//Registro/Velocidade").text # Assumindo que a velocidade está aqui
        frequencia_cardiaca = root.find(".//Registro/FrequenciaCardiaca").text # Frequência Cardíaca

        if taxa_amostragem_str is None:
            raise ValueError("Attribute 'TaxaAmostragem' not found in 'Registros' tag.")
        if sensibilidade_str is None:
            raise ValueError("Attribute 'Sensibilidade' not found in 'Registros' tag.")

        taxa_amostragem = float(re.search(r'\d+', taxa_amostragem_str).group())
        sensibilidade_valor = 0.005  # mV por unidade de amostra (5 microvolts = 0.005 mV)

        ecg_data = {}
        canais_ordenados = ["DI", "DII", "DIII", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        # Mapeia os canais encontrados no XML
        xml_canais = {canal.get("Nome"): canal for canal in root.findall(".//Canal")}

        for nome_canal in canais_ordenados:
            canal = xml_canais.get(nome_canal)
            if canal is None:
                print(f"Warning: Channel '{nome_canal}' not found in XML. Skipping.")
                ecg_data[nome_canal] = np.array([]) # Adiciona canal vazio para manter ordem
                continue

            amostras_str_elem = canal.find("Amostras")
            if amostras_str_elem is None or amostras_str_elem.text is None:
                print(f"Warning: 'Amostras' data missing for channel '{nome_canal}'. Skipping.")
                ecg_data[nome_canal] = np.array([])
                continue
            
            raw_samples_text = amostras_str_elem.text.replace('\r', '').replace('\n', '')
            try:
                raw_samples = np.array([float(x) for x in raw_samples_text.split(';') if x.strip()])
            except ValueError as ve:
                print(f"Error converting samples for channel '{nome_canal}': {ve}. Raw data: {raw_samples_text[:100]}...")
                ecg_data[nome_canal] = np.array([])
                continue

            samples_mv = raw_samples * sensibilidade_valor
            ecg_data[nome_canal] = samples_mv

        # Remove canais vazios ou não encontrados se necessário, ou lida com eles no plot
        ecg_data_validos = {k: v for k, v in ecg_data.items() if v.size > 0}
        if not ecg_data_validos:
            raise ValueError("No valid ECG data found after parsing channels.")

        # Informações do Paciente
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

        # Criação do Eixo do Tempo
        num_amostras = len(list(ecg_data_validos.values())[0])
        eixo_tempo = np.arange(num_amostras) / taxa_amostragem

        # --- Plotagem com Layout de ECG Padrão (3 colunas + ritmo) ---
        fig = plt.figure(figsize=(18, 12)) # Tamanho da figura ajustado para layout de 12 derivações
        
        # Calcula limites globais do eixo Y
        all_samples_values = np.concatenate(list(ecg_data_validos.values()))
        ymin_global = np.floor(all_samples_values.min() / 0.5) * 0.5
        ymax_global = np.ceil(all_samples_values.max() / 0.5) * 0.5
        if abs(ymax_global - ymin_global) < 1.0: # Garante uma faixa mínima de 1mV
            mid_point = (ymax_global + ymin_global) / 2
            ymin_global = mid_point - 0.5
            ymax_global = mid_point + 0.5
            
        # Adiciona um pequeno padding para não cortar o sinal
        ymin_global -= 0.1
        ymax_global += 0.1

        gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.7]) # 4 linhas para derivações, 1 para ritmo longa

        lead_order = ["DI", "DII", "DIII", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        # Plot das 12 derivações principais (4 linhas x 3 colunas)
        for idx, lead_name in enumerate(lead_order):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            samples_mv = ecg_data.get(lead_name, np.array([])) # Pega dados, ou array vazio se não existir
            if samples_mv.size > 0:
                ax.plot(eixo_tempo, samples_mv, linewidth=0.7, color='black')
                
                # Pulso de calibração (1mV por 0.2s = 25mm em 25mm/s)
                # Este pulso deve estar no início do gráfico
                calib_start_time = 0.05 # Início do pulso em segundos
                calib_end_time = calib_start_time + 0.2 # Duração de 0.2s
                calib_height = 1.0 # Altura de 1mV

                # Encontra índices correspondentes ao tempo
                idx_calib_start = np.argmin(np.abs(eixo_tempo - calib_start_time))
                idx_calib_end = np.argmin(np.abs(eixo_tempo - calib_end_time))
                
                # Cria a forma do pulso. Posição y pode ser ajustada para ficar visível.
                # Por exemplo, deslocado do zero para cima ou para baixo
                pulse_y_offset = ymin_global + (ymax_global - ymin_global) * 0.1 # 10% acima do min
                
                ax.plot([calib_start_time, calib_start_time], [pulse_y_offset, pulse_y_offset + calib_height], color='blue', linewidth=1.5)
                ax.plot([calib_start_time, calib_end_time], [pulse_y_offset + calib_height, pulse_y_offset + calib_height], color='blue', linewidth=1.5)
                ax.plot([calib_end_time, calib_end_time], [pulse_y_offset + calib_height, pulse_y_offset], color='blue', linewidth=1.5)
                
            ax.set_ylabel(f"{lead_name}", fontsize=10, rotation=0, ha='right') # Rótulo no início da linha
            ax.tick_params(axis='both', which='both', labelsize=8)
            ax.set_ylim(ymin_global, ymax_global) # Aplica limites globais do Y

            # Esconde tick labels para subplots não-finais
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Grade de ECG
            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.2))
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.01, 0.5))
            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.04), minor=True)
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.001, 0.1), minor=True)
            ax.grid(True, which='major', color='red', linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', color='pink', linestyle='-', linewidth=0.5)
            
            # Remover bordas desnecessárias
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Plot da derivação de Ritmo (e.g., DII longa)
        ax_rhythm = fig.add_subplot(gs[4, :]) # Ocupa todas as 3 colunas na última linha
        rhythm_lead_name = "DII" # Ou outro canal de ritmo se especificado
        rhythm_samples = ecg_data.get(rhythm_lead_name, np.array([]))
        if rhythm_samples.size > 0:
            ax_rhythm.plot(eixo_tempo, rhythm_samples, linewidth=0.7, color='black')
        ax_rhythm.set_ylabel(f"Ritmo ({rhythm_lead_name})", fontsize=10, rotation=0, ha='right')
        ax_rhythm.set_xlabel("Tempo (s)", fontsize=10)
        ax_rhythm.tick_params(axis='both', which='major', labelsize=8)
        ax_rhythm.set_ylim(ymin_global, ymax_global) # Aplica limites globais do Y

        # Grade de ECG para ritmo
        ax_rhythm.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.2))
        ax_rhythm.set_yticks(np.arange(ymin_global, ymax_global + 0.01, 0.5))
        ax_rhythm.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.04), minor=True)
        ax_rhythm.set_yticks(np.arange(ymin_global, ymax_global + 0.001, 0.1), minor=True)
        ax_rhythm.grid(True, which='major', color='red', linestyle='-', linewidth=0.8)
        ax_rhythm.grid(True, which='minor', color='pink', linestyle='-', linewidth=0.5)
        
        # Remover bordas desnecessárias
        ax_rhythm.spines['top'].set_visible(False)
        ax_rhythm.spines['right'].set_visible(False)
        ax_rhythm.spines['bottom'].set_visible(False)
        ax_rhythm.spines['left'].set_visible(False)


        # Adicionar informações de calibração/velocidade/FC no canto inferior esquerdo/direito
        fig.text(0.05, 0.01, f"Velocidade: {velocidade_exame} mm/s", fontsize=10, ha='left', va='bottom')
        fig.text(0.15, 0.01, f"Sensibilidade: {sensibilidade_valor*1000:.0f} µV/mm (10mm/mV)", fontsize=10, ha='left', va='bottom') # 10mm/mV é o padrão, então 1mm = 0.1mV. Se 5µV = 1 unidade, e Ganho é 10, então 10mm/mV
        fig.text(0.28, 0.01, f"FC: {frequencia_cardiaca} bpm", fontsize=10, ha='left', va='bottom')


        # Ajusta o layout geral
        plt.tight_layout(rect=[0.0, 0.05, 1, 0.96]) # Mais espaço no topo para o título e no fundo para as informações

        # Salva o gráfico em um buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close(fig) # Fecha a figura para liberar memória
        return buf

    except ET.ParseError as e:
        print(f"Erro de parsing XML: {e}")
        # Tentar extrair o contexto do erro
        match = re.search(r'line (\d+), column (\d+)', str(e))
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            lines = xml_content.splitlines()
            print("\n--- XML Content around error ---")
            start_line = max(0, line_num - 3)
            end_line = min(len(lines), line_num + 2)
            for i in range(start_line, end_line):
                print(f"Line {i+1}: {lines[i]}")
                if i + 1 == line_num:
                    print(" " * (col_num - 1) + "^ Error here")
            print("--------------------------------")
        return None
    except ValueError as e:
        print(f"Erro nos dados do XML ou metadados ausentes/inválidos: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None


# --- Rotas do Flask (permanecem as mesmas) ---

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
        # Tenta decodificar o arquivo. Se for .wxml, pode precisar de 'latin-1' ou similar.
        # Por padrão, vamos tentar 'utf-8'. Se falhar, pode ser um problema do arquivo.
        try:
            xml_content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Tenta outra codificação comum se UTF-8 falhar
            file.seek(0) # Volta ao início do arquivo
            xml_content = file.read().decode('latin-1') 
        
        image_buffer = gerar_ecg_do_xml_interno(xml_content)
        
        if image_buffer:
            return send_file(image_buffer, mimetype='image/png', as_attachment=False)
        else:
            return "Erro ao gerar o gráfico de ECG. Verifique o formato do XML e os metadados.", 500

if __name__ == '__main__':
    app.run(debug=True)