from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import logging
import random
import time
import re
import markdown
import html
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuración para subida de archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CHUNK_SIZE'] = 10000  # Tamaño aproximado de cada fragmento en tokens
app.config['REQUEST_TIMEOUT'] = 300  # Timeout aumentado a 5 minutos

OLLAMA_API_URL = 'http://localhost:11434/api/generate'

# Configurar reintentos y timeout
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)
http.request = lambda *args, **kwargs: requests.Session.request(http, *args, **{**kwargs, 'timeout': app.config['REQUEST_TIMEOUT']})

# Emojis simplificados
THINKING_EMOJI = '🤔'
RESPONSE_EMOJI = '🤖'
ERROR_EMOJI = '⚠️'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        total_pages = doc.page_count
        logging.info(f"Procesando PDF con {total_pages} páginas")
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            text += page_text
            logging.debug(f"Página {page_num}/{total_pages} procesada")
        
        doc.close()
        logging.info(f"PDF procesado completamente. Texto extraído: {len(text)} caracteres")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size):
    """Divide el texto en fragmentos de tamaño aproximado."""
    words = text.split()
    total_words = len(words)
    chunks = []
    current_chunk = []
    current_size = 0
    
    logging.info(f"Procesando texto de {total_words} palabras para fragmentación")
    
    for word in words:
        word_size = len(word.split())  # Aproximación simple de tokens
        if current_size + word_size > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            logging.debug(f"Fragmento creado con {len(current_chunk)} palabras")
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    # Asegurarse de incluir el último fragmento
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
        logging.debug(f"Último fragmento creado con {len(current_chunk)} palabras")
    
    total_words_in_chunks = sum(len(chunk.split()) for chunk in chunks)
    logging.info(f"Total de palabras procesadas: {total_words_in_chunks} de {total_words}")
    
    if total_words_in_chunks < total_words:
        logging.warning(f"Se perdieron {total_words - total_words_in_chunks} palabras durante el procesamiento")
    
    return chunks

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logging.info(f"Procesando archivo PDF: {filename}")
            
            # Extraer texto del PDF
            text = extract_text_from_pdf(file_path)
            if text:
                # Dividir el texto en fragmentos
                chunks = chunk_text(text, app.config['MAX_CHUNK_SIZE'])
                
                # Guardar los fragmentos en la sesión
                if 'pdf_chunks' not in app.config:
                    app.config['pdf_chunks'] = {}
                app.config['pdf_chunks'][filename] = chunks
                
                # Eliminar el archivo después de extraer el texto
                os.remove(file_path)
                
                logging.info(f"PDF procesado exitosamente. Generados {len(chunks)} fragmentos")
                
                return jsonify({
                    'success': True,
                    'message': f'PDF procesado exitosamente: {filename}',
                    'filename': filename,
                    'num_chunks': len(chunks)
                })
            else:
                return jsonify({'error': 'No se pudo extraer texto del PDF'}), 400
                
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

def clean_math_expressions(text):
    """Limpia y formatea expresiones matemáticas."""
    # No eliminar los backslashes necesarios para LaTeX
    replacements = {
        r'\\begin\{align\*?\}': '',
        r'\\end\{align\*?\}': '',
        r'\\begin\{equation\*?\}': '',
        r'\\end\{equation\*?\}': '',
        r'\\ ': ' '  # Reemplazar \\ espacio con un espacio normal
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def format_math(text):
    """Formatea expresiones matemáticas para KaTeX."""
    def process_math_content(match):
        content = match.group(1).strip()
        content = clean_math_expressions(content)
        return f'$${content}$$'

    # Procesar comandos especiales de LaTeX antes de los bloques matemáticos
    text = re.sub(r'\\boxed\{\\text\{([^}]*)\}\}', r'<div class="boxed">\1</div>', text)
    text = re.sub(r'\\boxed\{([^}]*)\}', r'<div class="boxed">\1</div>', text)
    
    # Procesar bloques matemáticos inline y display
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: f'${m.group(1)}$', text)
    text = re.sub(r'\\\[(.*?)\\\]', process_math_content, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: f'${m.group(1)}$', text)
    
    # Preservar comandos LaTeX específicos
    text = re.sub(r'\\times(?![a-zA-Z])', r'\\times', text)  # Preservar \times
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\\frac{\1}{\2}', text)  # Preservar fracciones
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)  # Manejar \text correctamente
    
    return text

def format_code_blocks(text):
    """Formatea bloques de código con resaltado de sintaxis."""
    def replace_code_block(match):
        language = match.group(1) or 'plaintext'
        code = match.group(2).strip()
        return f'```{language}\n{code}\n```'

    # Procesar bloques de código
    text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
    return text

def format_response(text):
    """Formatea la respuesta completa con soporte para markdown, código y matemáticas."""
    # Primero formatear expresiones matemáticas
    text = format_math(text)
    
    # Formatear bloques de código
    text = format_code_blocks(text)
    
    # Convertir markdown a HTML preservando las expresiones matemáticas
    # Escapar temporalmente las expresiones matemáticas
    math_blocks = []
    def math_replace(match):
        math_blocks.append(match.group(0))
        return f'MATH_BLOCK_{len(math_blocks)-1}'

    # Guardar expresiones matemáticas
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', math_replace, text, flags=re.DOTALL)
    
    # Convertir markdown a HTML
    md = markdown.Markdown(extensions=['fenced_code', 'tables'])
    text = md.convert(text)
    
    # Restaurar expresiones matemáticas
    for i, block in enumerate(math_blocks):
        text = text.replace(f'MATH_BLOCK_{i}', block)
    
    # Limpiar y formatear el texto
    text = text.replace('</think>', '').replace('<think>', '')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    
    return text.strip()

def decorate_message(message, is_error=False):
    """Decora el mensaje con emojis y formato apropiado."""
    emoji = ERROR_EMOJI if is_error else RESPONSE_EMOJI
    if is_error:
        return f"{emoji} {message}"
    
    formatted_message = format_response(message)
    return f"{emoji} {formatted_message}"

def get_thinking_message():
    """Genera un mensaje de 'pensando' aleatorio."""
    messages = [
        "Analizando tu pregunta...",
        "Procesando la información...",
        "Elaborando una respuesta...",
        "Pensando...",
        "Trabajando en ello...",
    ]
    return f"{THINKING_EMOJI} {random.choice(messages)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    model = data.get('model', 'deepseek-r1:7b')
    filename = data.get('pdf_file', None)
    chunk_index = data.get('chunk_index', 0)
    
    app.logger.debug(f"Mensaje recibido: {user_message}")
    app.logger.debug(f"Modelo seleccionado: {model}")

    def generate():
        try:
            # Enviar mensaje inicial de "pensando"
            thinking_msg = get_thinking_message()
            yield json.dumps({
                'thinking': thinking_msg
            }) + '\n'
            
            # Preparar el prompt base
            prompt = user_message
            
            # Solo incluir contexto del PDF si hay un archivo activo Y está en el mismo chat
            if filename and filename in app.config.get('pdf_chunks', {}) and data.get('isPdfChat', False):
                chunks = app.config['pdf_chunks'][filename]
                
                # Construir el contexto combinando fragmentos relevantes
                context_chunks = []
                
                # Siempre incluir el fragmento actual
                current_chunk = chunks[chunk_index]
                context_chunks.append(f"Fragmento {chunk_index + 1}:\n{current_chunk}")
                
                # Incluir fragmentos adyacentes si están disponibles
                if chunk_index > 0:
                    prev_chunk = chunks[chunk_index - 1]
                    context_chunks.insert(0, f"Fragmento {chunk_index}:\n{prev_chunk}")
                
                if chunk_index < len(chunks) - 1:
                    next_chunk = chunks[chunk_index + 1]
                    context_chunks.append(f"Fragmento {chunk_index + 2}:\n{next_chunk}")
                
                # Combinar los fragmentos en un solo contexto
                combined_context = "\n\n".join(context_chunks)
                
                prompt = f"""Contexto del PDF (fragmentos {chunk_index + 1} y adyacentes de {len(chunks)} totales):

{combined_context}

Pregunta del usuario:
{user_message}

Por favor, responde la pregunta basándote en el contenido proporcionado del PDF.
Si la respuesta podría estar en otros fragmentos no incluidos, indícalo y sugiere revisar otros fragmentos."""
            
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': True
            }
            
            app.logger.debug(f"Enviando solicitud a Ollama API con payload: {payload}")
            
            try:
                response = http.post(
                    OLLAMA_API_URL,
                    json=payload,
                    stream=True,
                    timeout=60  # Aumentar timeout a 60 segundos
                )
            except requests.exceptions.Timeout:
                error_msg = "La solicitud está tomando más tiempo de lo esperado. Por favor, intenta con un mensaje más corto o espera un momento."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            except requests.exceptions.ConnectionError:
                error_msg = "No se pudo conectar con Ollama. Por favor, verifica que Ollama esté corriendo."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            
            app.logger.debug(f"Estado de respuesta de Ollama API: {response.status_code}")
            if response.status_code != 200:
                error_msg = f"Error al conectar con Ollama API. Código de estado: {response.status_code}. Respuesta: {response.text}"
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return

            # Limpiar mensaje de "pensando" y comenzar a mostrar la respuesta
            yield json.dumps({'clear_thinking': True}) + '\n'
            
            # Inicializar acumulador de respuesta
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        app.logger.debug(f"Fragmento de respuesta recibido: {json_response}")
                        ai_response = json_response.get('response', '')
                        if ai_response:
                            full_response += ai_response
                            # Formatear y enviar la respuesta completa hasta el momento
                            decorated_response = decorate_message(full_response)
                            yield json.dumps({'response': decorated_response}) + '\n'
                        
                    except json.JSONDecodeError as e:
                        app.logger.error(f"Error al decodificar JSON: {str(e)} para la línea: {line}")
                        continue

        except Exception as e:
            error_msg = f"Error de conexión: {str(e)}"
            app.logger.error(error_msg)
            yield json.dumps({
                'error': decorate_message(error_msg, is_error=True)
            }) + '\n'

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    status = {
        'status': 'healthy',
        'message': "Servidor en funcionamiento",
        'timestamp': time.time()
    }
    return json.dumps(status)

if __name__ == '__main__':
    app.logger.info("\n=== Servidor de Chat IA Iniciado ===")
    app.run(debug=True, port=5000) 