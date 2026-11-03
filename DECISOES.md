# Decisões de engenharia — ClipRush Gamer AI v3 Workspace

## Escopo

Aplicação local Python/Streamlit para analisar gameplays e criar Shorts/TikToks 9:16 sem webcam, com fila SQLite, configurações persistentes, resultados/editor, kit de postagem, calendário, métricas, checklist, biblioteca JSON, geração automática e aba Histórico. O app é CPU-first e não publica nem agenda em contas externas.

## Desempenho e persistência

- Alvo: Windows 10/11, Python 3.11 x64, Ryzen 5 e 16 GB. FFmpeg/ffprobe são externos; render e transcrição permanecem CPU-only. Ollama é um serviço local opcional.
- A fila tem worker independente e serializa jobs e clipes. A inferência reduz keyframes a 320×180 por painel, agrupa até três janelas e limita context/thread; não carrega vídeo longo em RAM.
- A IA visual processa até 18 candidatos por gameplay, distribuídos pela timeline, em até seis lotes. É amostragem orientada por picos/probes, não compreensão contínua de cada frame. Timestamps sugeridos são aproximados e validados contra duração/peak local.
- O cancelamento percorre JSONs de progresso aninhados, inclusive FFmpeg da renderização automática; assets já concluídos não são removidos.
- `queue_jobs`, posts, métricas, checklist e índice de renders ficam em SQLite/WAL. Settings e progressos são JSONs gravados atomicamente. Migrações são aditivas; o ZIP de atualização não contém banco, `.venv`, vídeos nem pastas de usuário.

## Análise de gameplay e papel da IA

- `analyzer.py` continua extraindo áudio/VAD, energia, movimento, HUD, probes de timeline inteira, fala contínua e heurísticas de cutscene. Os picos e medidas locais selecionam a amostra visual e permanecem como referência temporal.
- `ia_manager.py` verifica o inventário local, extrai três frames próximos por candidato, monta contact sheet 960×180, codifica em base64 e envia via `/api/chat` apenas a `http://localhost:11434`. Nenhum arquivo, caminho, vídeo inteiro, áudio bruto ou transcript entra no request.
- Preferência padrão: `llava:7b` por tamanho para o alvo de 16 GB. Se não estiver instalado, o manager detecta `llama3.2-vision:11b` ou outra tag compatível já instalada. Não baixa modelos; o usuário inicia `ollama pull` no computador. O inventário consultado previamente no notebook continha `llama3.2:3b`, mas não um VLM.
- A resposta usa JSON Schema estrito; classe visual é limitada a ação/cutscene/menu-carregamento/travessia/desconhecido. Score final de candidatos com visão combina 65% score visual do modelo, 25% ranking local e 10% sinal local adicional; cutscenes usam componentes de fala/música/tensão e não ganham score por cortes de câmera.
- Os limites `moment_start/end_offset` são advisory, limitados a ±18 s, checados e guardados como evidência. O pico local não é substituído por um offset do modelo. Menus/carregamentos e travessias são rebaixados/descartados apenas com baixa relevância/score; cutscenes permanecem no modo padrão. A lista final continua limitada por modo/participação máxima de cenas.
- Fallback hierárquico: (1) VLM local quando instalado e válido; (2) `llama3.2:3b` classifica apenas sinais numéricos locais e é rotulado como **sem visão**; (3) motor heurístico local se Ollama/JSON/timeout falhar. Nenhum fallback diz que viu quadros. A disponibilidade de modelos integra a chave de cache para nova análise após instalação.
- O score 0–100 é relativo e não representa probabilidade garantida de viralização. VLM não pode inventar headshot, arma, eliminação ou outro evento específico; títulos específicos exigem evidência forte/confirmação manual. Reações de cutscene permanecem neutras.

## Ollama editorial e texto

- O modelo de texto padrão é `llama3.2:3b`, limitado a localhost, CPU (`num_gpu=0`), quatro threads e saída JSON validada. Títulos devem ter curiosidade/tensão e ser distintos, sem clickbait falso; tags confiáveis do kit local são preservadas.
- Cada geração passa por validação de contrato e filtro de eventos específicos. Falha ou conteúdo arriscado volta ao kit local. O gerador editorial recebe `ai_evidence` apenas quando disponível e nunca usa caminho de arquivo.
- Todos os textos continuam editáveis e persistidos por análise/corte/plataforma. Nenhuma edição de metadados altera silenciosamente pixels já renderizados; re-render é explícito.

## Render automático e Histórico

- Após a análise, se ativado, `_auto_render_analysis` processa os 5–10 clipes selecionados um por vez, em saída final 1080×1920, com composição blur + gameplay completa, capa e legendas opcionais.
- Para cada asset, grava registro em `rendered_assets`, `clip_content` e `posts`, incluindo título, descrição, hashtags, comentário, score, arquivo e associação ao job. Falhas isoladas são registradas no resultado do job e não apagam renders já concluídos.
- Aba **Histórico** lista posts e permite abrir/pré-visualizar/baixar o MP4, consultar capa, copiar textos e abrir as páginas de upload. Os links abrem YouTube Studio/TikTok, mas o browser não pode selecionar um arquivo local por URL: seleção e confirmação do upload são manuais.
- Auto-render é serial e pode demorar em CPU; é configurável. Modelos visuais de 7B+ podem ter latência/memória significativa apesar de `num_gpu=0`; preferimos amostrar e descarregar o modelo ao final.

## Renderização e UX

- Saída libx264 vertical, gameplay 16:9 inteira e HUD preservada sobre fundo desfocado; zoom dinâmico com easing atua apenas no fundo blur, mantendo o jogo central sem crop. Removido flash branco; composição (incluindo texto) recebe fade in preto de 0,2 s desde o primeiro frame.
- Capa JPEG é extraída do MP4 já finalizado, selecionando o primeiro frame com PTS >= +1,0 s do arquivo renderizado. Isso inclui o reframing/textos/fade final, não depende de buscar o timestamp `peak` no vídeo fonte e evita frames claros/brancos da rotina antiga. A extração valida o arquivo e falha com erro legível se não houver imagem útil.
- Teaser, safe zones, legenda Whisper tiny e reação neutra de cutscene continuam disponíveis.
- A navegação mantém abas de fila, resultados, kit, calendário, métricas, checklist, biblioteca e configurações; adiciona Histórico. No card do corte existem selecionar/anterior/próximo para evitar voltar ao seletor superior.
- Sidebar mostra FFmpeg, Ollama texto/visão e espaço em disco. Status indica claramente se visão existe e se caiu para fallback.

## Testes e limites

- Testes cobrem fallback seguro, schema JSON, IDs opacos sem path, detecção alternativa do modelo, escolha distribuída pela timeline, extração FFmpeg de contato de vídeo sintético, editorial seguro, auto-render e persistência SQLite.
- HUD/cutscene, VLM e ranking ainda podem errar, especialmente em ação rápida, HUD variável ou compressão alta. Revisão de título, spoiler, direitos, legenda e corte final é necessária antes da publicação.
- AppTest e suíte rodam no Sandbox Linux; o PC Windows deve confirmar FFmpeg/PATH, modelos Ollama, permissões e tempo de render.
