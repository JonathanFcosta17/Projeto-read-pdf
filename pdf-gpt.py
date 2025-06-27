import streamlit as st
import openai
import base64
import json
import pandas as pd
import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract
import re
from typing import Optional
import io
from datetime import datetime
import fitz  # PyMuPDF


class PDFTextExtractor:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def pdf_to_images(self, pdf_path, dpi=300):
        return convert_from_path(pdf_path, dpi=dpi)

    def pdf_to_images_pymupdf(self, pdf_path):
        images = []
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
        return images

    def enhance_image_for_osd(self, image: Image.Image) -> Image.Image:
        """Melhora a qualidade da imagem para melhor detecção OSD"""
        if image.mode != 'L':
            image = image.convert('L')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        return image

    def detect_orientation_osd(self, image: Image.Image, max_attempts: int = 3) -> tuple[Optional[int], Optional[float]]:
        """
        Detecta orientação usando Tesseract OSD
        Retorna (ângulo de rotação, confiança)
        """
        for attempt in range(1, max_attempts + 1):
            try:
                enhanced_image = self.enhance_image_for_osd(image)
                osd_config = '--psm 0 -c min_characters_to_try=5'
                osd_data = pytesseract.image_to_osd(enhanced_image, config=osd_config)
                
                orientation_match = re.search(r'Orientation in degrees: (\d+)', osd_data)
                rotate_match = re.search(r'Rotate: (\d+)', osd_data)
                confidence_match = re.search(r'Orientation confidence: ([\d.]+)', osd_data)
                
                if orientation_match and rotate_match:
                    detected_orientation = int(orientation_match.group(1))
                    rotation_needed = int(rotate_match.group(1))
                    confidence = float(confidence_match.group(1)) if confidence_match else 0
                    
                    if confidence < 0.4:
                        continue
                    return rotation_needed, confidence
                else:
                    return None, None

            except Exception as e:
                st.error(f"Erro na detecção OSD: {e}")
                return None, None
        
        return None, None

    def correct_image_orientation(self, image: Image.Image) -> tuple[Image.Image, Optional[float]]:
        """
        Detecta e corrige automaticamente a orientação da imagem
        Retorna (imagem corrigida, confiança)
        """
        rotation_angle, confidence = self.detect_orientation_osd(image)
        
        if rotation_angle is not None:
            if rotation_angle == 0:
                return image, confidence
            else:
                corrected_image = image.rotate(-rotation_angle, expand=True)
                return corrected_image, confidence
        else:
            return image, None


class GPTImageTableExtractor:
    def __init__(self, image_path, openai_api_key):
        self.image_path = image_path
        self.client = openai.OpenAI(api_key=openai_api_key)

    def image_to_base64(self):
        with open(self.image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def call_gpt_api(self, prompt):
        base64_img = self.image_to_base64()
        image_payload = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}"
            }
        }

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    image_payload,
                    {"type": "text", "text": prompt}
                ]}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def get_extraction_prompt(self):
        return '''
Você é um especialista em extração de dados de tabelas em imagens.
Sua tarefa é analisar a imagem fornecida e extrair, escrevendo um json, SOMENTE os dados tabulares contidos nela sem explicações, comentários ou instruções. 
A imagem contém uma tabela com vários valores em várias colunas, porém você deve extrair apenas as colunas: Cliente, Emissão, Vencimento, Valor e Saldo.
O resultado dever ser organizado em formato JSON, onde cada linha da tabela é um objeto com as chaves correspondentes às colunas.
Aqui está um exemplo de como o JSON deve ser estruturado:
[
    {
        "Cliente": "AGUILERA AUTOPECAS DE GOIAS LTDA",
        "Emissão": "28/03/2025",
        "Vencimento": "27/04/2025",
        "Valor": 360.00,
        "Saldo": 360.00
    }
]

IMPORTANTE: Retorne APENAS o JSON válido, sem texto adicional, sem markdown, sem explicações.'''

    def extract_table_data(self):
        """Extrai dados da tabela e retorna como DataFrame"""
        prompt = self.get_extraction_prompt()
        extracted_text = self.call_gpt_api(prompt)

        try:
            cleaned_text = extracted_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text.replace('```', '').strip()

            start_list = cleaned_text.find('[')
            end_list = cleaned_text.rfind(']') + 1
            
            if start_list == -1 or end_list == 0:
                start_obj = cleaned_text.find('{')
                end_obj = cleaned_text.rfind('}') + 1
                if start_obj != -1 and end_obj != 0:
                    json_str = cleaned_text[start_obj:end_obj]
                    json_data = [json.loads(json_str)]
                else:
                    raise ValueError("Não foi possível encontrar JSON válido na resposta")
            else:
                json_str = cleaned_text[start_list:end_list]
                json_data = json.loads(json_str)

            df = pd.DataFrame(json_data)
            
            colunas_desejadas = ["Cliente", "Emissão", "Vencimento", "Valor", "Saldo"]
            colunas_existentes = [col for col in colunas_desejadas if col in df.columns]
            
            if colunas_existentes:
                df = df[colunas_existentes]
            
            return df

        except Exception as e:
            st.error(f"Erro ao interpretar resposta do GPT: {e}")
            return pd.DataFrame()


def to_excel(df):
    """Converte DataFrame para Excel em memória"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados Extraídos')
    output.seek(0)
    return output


def get_config():
    """Obtém configurações de forma segura"""
    try:
        # Tenta primeiro os secrets do Streamlit (para deploy)
        openai_key = st.secrets["OPENAI_API_KEY"]
        tesseract_path = st.secrets.get("TESSERACT_PATH", "tesseract")
    except:
        # Fallback para variáveis de ambiente ou valores padrão (desenvolvimento)
        openai_key = os.getenv("OPENAI_API_KEY", "sua-chave-aqui")
        tesseract_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    
    return openai_key, tesseract_path


def main():
    st.set_page_config(
        page_title="Extrator de Dados PDF",
        page_icon="📄",
        layout="wide"
    )
    
    # Obtém configurações
    try:
        OPENAI_API_KEY, TESSERACT_PATH = get_config()
        if not OPENAI_API_KEY or OPENAI_API_KEY == "sua-chave-aqui":
            st.error("❌ API Key do OpenAI não configurada!")
            st.info("Configure a variável OPENAI_API_KEY nos secrets ou variáveis de ambiente.")
            return
    except Exception as e:
        st.error(f"❌ Erro na configuração: {e}")
        return
    
    st.title("📄 Extrator de Dados de PDF")
    st.markdown("---")
    
    st.markdown("### 📋 Colunas Extraídas")
    st.info("O sistema extrai apenas as colunas:\n- Cliente\n- Emissão\n- Vencimento\n- Valor\n- Saldo")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Selecione o arquivo PDF",
        type=['pdf'],
        help="Faça upload do arquivo PDF contendo as tabelas"
    )
    
    if uploaded_file is not None:
        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        # Botão para processar
        if st.button("🚀 Processar PDF", type="primary"):
            try:
                with st.spinner("Processando PDF..."):
                    # Inicializa o extrator
                    extractor = PDFTextExtractor(TESSERACT_PATH if TESSERACT_PATH != "tesseract" else None)
                    
                    # Converte PDF para imagens
                    st.info("📄 Convertendo PDF para imagens...")
                    pages = extractor.pdf_to_images_pymupdf(pdf_path)  # <-- Troque para esta linha!
                    st.info(f"{len(pages)} páginas encontradas.")
                    
                    if not pages:
                        st.error("❌ Não foi possível converter o PDF em imagens.")
                        return
                    
                    # Processa as páginas
                    all_dfs = []
                    progress_bar = st.progress(0)
                    
                    for idx, page in enumerate(pages):
                        progress_bar.progress((idx + 1) / len(pages))
                        st.info(f"Processando página {idx+1}...")
                        
                        # Salva imagem temporária
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                            page.save(tmp_img.name)
                            tmp_img_path = tmp_img.name

                        try:
                            gpt_extractor = GPTImageTableExtractor(tmp_img_path, OPENAI_API_KEY)
                            df = gpt_extractor.extract_table_data()
                            if not df.empty:
                                df['Página'] = idx + 1
                                all_dfs.append(df)
                                st.success(f"Página {idx+1}: {len(df)} registros extraídos.")
                            else:
                                st.warning(f"Página {idx+1}: Nenhum dado extraído.")
                        except Exception as e:
                            st.error(f"Erro ao processar página {idx+1}: {e}")

                        # Remove arquivo temporário
                        try:
                            os.unlink(tmp_img_path)
                        except Exception as e:
                            st.warning(f"Não foi possível remover o arquivo temporário: {e}")
                    
                    # Combina todos os DataFrames
                    if all_dfs:
                        final_df = pd.concat(all_dfs, ignore_index=True)
                        
                        # Reordena colunas
                        cols = ['Página'] + [col for col in final_df.columns if col != 'Página']
                        final_df = final_df[cols]
                        
                        st.success(f"✅ Extração concluída! Extraídos {len(final_df)} registros de {len(pages)} páginas.")
                        
                        # Exibe a tabela
                        st.markdown("### 📊 Dados Extraídos")
                        st.dataframe(final_df, use_container_width=True)
                        
                        # Botão para download
                        excel_data = to_excel(final_df)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"dados_extraidos_{timestamp}.xlsx"
                        
                        st.download_button(
                            label="📥 Baixar Excel",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Estatísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Registros", len(final_df))
                        with col2:
                            st.metric("Páginas Processadas", len(pages))
                        with col3:
                            if 'Valor' in final_df.columns:
                                total_valor = final_df['Valor'].sum()
                                st.metric("Valor Total", f"R$ {total_valor:,.2f}")
                    
                    else:
                        st.error("❌ Nenhum dado foi extraído do PDF.")
                        
            except Exception as e:
                st.error(f"❌ Erro durante o processamento: {e}")
                st.exception(e)
            finally:
                # Remove arquivo temporário
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
    else:
        st.info("👆 Faça upload de um arquivo PDF para começar.")


if __name__ == "__main__":
    main()