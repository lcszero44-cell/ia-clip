# Ollama Vision — referência e implementação ClipRush

## Contrato oficial

A API REST `/api/chat` recebe imagens de `messages[].images` como strings base64, junto com texto no mesmo request: [Ollama Vision](https://docs.ollama.com/capabilities/vision). O campo `format` aceita `json` ou um JSON Schema; a documentação recomenda incluir o schema no contexto e usar temperatura baixa para restringir a forma da resposta: [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs).

## Modelos considerados

- `llava:7b`: 4,7 GB, 32K de contexto, entrada de imagem e texto. Para Ryzen 5/16 GB foi definido como padrão pelo menor tamanho entre as opções consideradas.
- `llama3.2-vision:11b`: 7,8 GB, 128K de contexto e entrada multimodal. O modelo é significativamente maior e pode pressionar RAM/CPU. A página oficial informa que as tarefas imagem+texto são oficialmente suportadas em inglês; o pipeline formula a avaliação visual em inglês e faz a geração final de copy em `llama3.2:3b` PT-BR.
- Fontes da biblioteca oficial: [LLaVA](https://ollama.com/library/llava) e [Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision).

A consulta anterior a `/api/tags` no notebook informou `llama3.2:3b` instalado, sem `llava` nem `llama3.2-vision`. Portanto, a instalação de um modelo visual é necessária para a etapa multimodal real; o aplicativo não inicia esse download automaticamente.

## Uso no ClipRush

`ia_manager.py` procura o modelo configurado e, se não existir, reconhece uma tag `llava` ou `llama3.2-vision` que já esteja instalada. O pipeline extrai três quadros próximos ao pico de cada candidato, reduz cada um a 320×180, compõe uma imagem de 960×180 e envia lotes pequenos sequencialmente a `http://localhost:11434`. A amostragem é limitada a até 18 janelas por vídeo para reduzir memória/latência; não é inferência em todos os frames. A IA retorna score, classe visual, relevância e limites aproximados em schema JSON, validados no analyzer. Os limites são advisory e o score final inclui sinais locais.

Inferência visual usa `num_gpu=0`, `num_thread=4`, contexto limitado e `keep_alive=0` no último lote para descarregar o modelo. Em caso de ausência/erro/time-out do VLM, o fluxo tenta ranking textual do Llama 3.2 3B sobre métricas resumidas de áudio/VAD/movimento; se falhar, permanece no motor heurístico local. A interface identifica explicitamente se o modelo textual não viu os frames.

Somente keyframes reduzidos e medidas temporais são enviados ao loopback local. O vídeo completo, áudio bruto, caminho/nome do arquivo e transcrição não são incluídos no request; nenhum endpoint remoto é aceito pelo módulo de visão. O resultado pode ainda confundir uma ação rápida, menu, HUD variável ou cutscene; rever o corte final continua necessário.
