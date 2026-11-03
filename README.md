# ClipRush Gamer AI v3 — Workspace local com análise Ollama multimodal

Aplicação local em Python/Streamlit para encontrar momentos fortes de gameplays PS4/PC e transformá-los em Shorts/TikToks 9:16, sem webcam e preservando a tela de jogo/HUD 16:9. Esta versão mantém o workspace V3 (fila, edição, calendário, métricas, checklist e biblioteca) e acrescenta análise visual Ollama, re-ranking por IA, render automático em série e histórico persistente de vídeos e kits.

O perfil é CPU-first para Ryzen 5 multi-core e 16 GB de RAM. A fila processa um job por vez, limita threads e pode reduzir prioridade; a inferência de imagem é sequencial, com miniaturas pequenas. Duração e uso de CPU/RAM variam por codec, resolução, modelo e extensão da gameplay. O score é uma classificação relativa, nunca uma garantia de viralização ou alcance.

## Instalação no Windows 10/11

1. Instale **Python 3.11 x64** de <https://www.python.org/downloads/> e habilite o launcher `py.exe`.
2. Instale um build **FFmpeg full** e adicione a pasta `bin` ao `PATH`. Confirme em um Prompt novo:
   ```bat
   ffmpeg -version
   ffprobe -version
   py -3.11 --version
   ```
3. Extraia o ZIP para uma pasta local gravável, por exemplo `C:\ClipRush Gamer AI`.
4. Dê duplo clique em `abrir_app.bat`. O primeiro início cria `venv`, instala os requisitos, valida a biblioteca e inicia o Streamlit em `127.0.0.1`.
5. Para a linha de comando, execute `iniciar.bat` ou `executar.bat`. O painel abre com `abrir_app.bat`.

Dependências Python são instaladas pelo launcher. FFmpeg/ffprobe e Ollama são programas externos. Whisper tiny e Silero VAD são usados localmente. Não é necessário inserir credenciais de YouTube/TikTok; o app abre as páginas oficiais para envio assistido.

### Atualizar uma instalação existente

1. Feche o painel/terminal do ClipRush e faça uma cópia de segurança da pasta atual.
2. Extraia este ZIP sobre a **mesma pasta**, permitindo substituir os arquivos de código. O pacote não contém banco de dados, gameplays nem resultados e, portanto, não os sobrescreve.
3. Se você editou `biblioteca/biblioteca_editorial.json`, guarde uma cópia antes de extrair e restaure-a depois; o ZIP contém a biblioteca padrão.
4. Apague apenas a pasta antiga `venv` para que o launcher instale as versões declaradas no ZIP. Não apague `data`, `uploads`, `output`, `models` nem `temp` se deseja conservar preferências, fila, vídeos e análises.
5. Inicie com `abrir_app.bat`. Confirme antes que FFmpeg/ffprobe estejam no `PATH` e que a pasta do projeto permita gravação.

## IA local: análise visual, fallback e privacidade

O módulo `ia_manager.py` é chamado antes do ranking final. O app amostra candidatos distribuídos pela linha do tempo, extrai até 18 janelas em uma análise e 3 quadros por janela, reduzidos a um contact sheet de 960×180 (3 painéis de 320×180), e envia-os em lotes de até 3 ao **Ollama em `http://localhost:11434`**. O prompt inclui apenas IDs temporais e sinais resumidos locais (áudio, VAD, HUD/movimento); ele não inclui caminho/nome do arquivo, vídeo inteiro, áudio bruto ou transcrição. A inferência usa CPU (`num_gpu=0`, quatro threads), schema JSON, temperatura baixa e descarrega o modelo visual após o último lote.

O ranking visual combina a nota visual do modelo (65%), a pontuação local de sinais (25%) e uma medida local adicional (10%). A IA sugere classe visual (ação, cutscene, menu/carregamento, caminhada ou desconhecido), relevância e limites aproximados relativos ao pico. O analyzer valida os valores, limita os offsets e mantém seu pico local como referência. Cutscenes continuam válidas; não são penalizadas por cortes de câmera. A IA visual é conservadora e não transforma observação incerta em “headshot”, arma, eliminação, combo ou outro evento específico.

**Limite importante:** isso não é análise visual contínua de cada frame. Para desempenho local, o modelo vê keyframes de candidatos espalhados pela timeline, não todos os segundos da gravação; limites são estimativas para ajudar a compor o corte e devem ser revistos. Um VLM pode errar ou não reconhecer um evento rápido.

O padrão é `llava:7b`, escolhido pelo tamanho mais adequado a 16 GB; se estiver instalado, o app também pode detectar `llama3.2-vision:11b`. A consulta de inventário feita anteriormente no notebook encontrou `llama3.2:3b`, mas **nenhum modelo visual**. Para habilitar análise por visão, instale manualmente no próprio computador e reinicie/atualize o status:

```powershell
ollama pull llava:7b
```

Alternativa maior, que pode pressionar a memória num PC de 16 GB:

```powershell
ollama pull llama3.2-vision:11b
```

O aplicativo não baixa modelos automaticamente. Se o Ollama/modelo visual estiver indisponível, tenta `llama3.2:3b` para ranquear **somente métricas locais de áudio/VAD/movimento**, explicitamente sem visão. Se esse pedido também falhar, usa o ranking heurístico local; nenhum desses casos interrompe o app. Para gerar títulos, hooks, descrição, hashtags e comentário, o texto passa pelo modelo Llama textual, quando disponível, e é bloqueado se inventar eventos específicos não confirmados. Sem Ollama, o gerador editorial offline permanece disponível. A configuração e os limites de amostragem ficam em **Configurações → IA local · Ollama**.

O loopback `localhost` significa o serviço Ollama no mesmo computador onde o ClipRush está sendo executado. Miniaturas e sinais saem do processo do app somente para esse serviço local; não são enviados a um serviço de IA remoto. O modelo visual precisa ser instalado pelo usuário; essa instalação não foi iniciada pelo ClipRush.

## Fluxo automático

Na aba **Criar**, selecione o arquivo (ou caminho local), confirme a análise rápida/detalhada, plataforma, uso da IA e **Renderizar MP4 final + capa automaticamente**. Ao enfileirar, o worker local:

1. varre os sinais locais e escolhe de 5 a 10 clipes segundo duração e ranking;
2. tenta reclassificar os candidatos com visão Ollama e repondera score; usa fallback textual/local se necessário;
3. gera automaticamente o kit de postagem e renderiza, em série, MP4 final vertical, capa e legendas se configuradas;
4. salva os assets, texto e registro editorial no SQLite para as abas **Resultados**, **Kit** e **Histórico**.

O auto-render usa saída final em 1080×1920; pode levar vários minutos por gameplay, especialmente com 5–10 cortes e CPU-only. Pode ser desativado na criação ou em Configurações. Ao cancelar, os cortes já concluídos continuam salvos; o clipe em processamento recebe cancelamento cooperativo.

## Ingestão, fila e histórico

Na aba **Criar cortes**, envie MP4/MKV/MOV/M4V/AVI ou informe o caminho local. Para gameplays longas, prefira um caminho local, por exemplo `D:\Gameplays\sessao.mp4`, para evitar cópia/limite do upload do browser. Na aba **Fila**, é possível adicionar vários arquivos por upload ou caminhos locais, um por linha.

- A fila persistente fica em SQLite (`data/cliprush_local.db`) e é executada por um worker independente, em série.
- Reruns do Streamlit não cancelam análise/render. A fila atualiza progresso; pode cancelar job pendente ou ativo e reenfileirar falhos/cancelados.
- A análise e os assets são associados ao job, sessão, corte e plataforma. A aba **Histórico** lista cada MP4, capa e kit gerados, mesmo depois de processar outros vídeos.
- O histórico do banco não apaga arquivos quando se remove registro da fila.

## Resultados e edição

A aba **Resultados** localiza `output/sessoes/**/analysis.json`, inclusive sessões anteriores mantidas na pasta. Mostra score, tipo, sinais, modelo usado, evidência resumida e aviso quando o fallback não tinha visão. Use anterior/próximo ou selecione o corte no próprio card; cada clipe apresenta prévia/download quando já renderizado e editor persistente para alterar título de postagem, texto na tela, jogo, descrição, hashtags e comentário fixado.

O reframing preserva a gameplay 16:9 completa no centro (sem crop da HUD) sobre fundo vertical desfocado. O fundo recebe zoom dinâmico com aceleração/desaceleração suave nos momentos de impacto; a gameplay central e a HUD não são ampliadas nem cortadas. O vídeo começa com fade preto cinematográfico de 0,2 s; o flash branco agressivo foi removido. A capa é extraída do próprio MP4 final no primeiro frame disponível em +1,0 s, após o fade, evitando thumbnails brancas/cinzentas causadas por capturar antes da composição/render. O teaser do pico, cor/brilho seletivos, safe zones e legendas locais permanecem disponíveis. Em cutscene, a reação visual é neutra. Alterar texto não modifica pixels do MP4 já renderizado; use **Salvar e renderizar com este texto** para gerar uma nova versão.

Em cada corte, abra **Publicação rápida** ou a aba **Histórico** para copiar título, descrição e hashtags e abrir:

- [YouTube Studio — Upload](https://studio.youtube.com/channel/UC/videos/upload?d=ud)
- [TikTok Creator Center — Upload](https://www.tiktok.com/creator-center/upload)

Por segurança, o navegador não permite que uma URL anexe ou pré-selecione automaticamente um arquivo local. A página abre com a sessão da conta se já estiver autenticada; selecione o MP4 e confirme o envio manualmente. O ClipRush não publica nem agenda em contas externas.

## Kit, calendário, biblioteca e métricas

A aba **Kit** oferece três títulos, hook, descrição, CTA, hashtags, comentário fixado e texto de reação. A geração usa a evidência visual validada quando disponível e não afirma eventos que o classificador não confirmou. Títulos, edição e conteúdo do post são persistidos por análise/corte/plataforma.

O **Calendário editorial** guarda rascunho, roteiro, agendamento local e status de publicação. **Métricas** e **Checklist** são manuais; não há conexão para ler analytics de contas.

`biblioteca/biblioteca_editorial.json` contém 239 itens em dez categorias. A aba **Biblioteca** permite editar/adicionar/remover linhas e valida unicidade, plataformas, total mínimo de 200 itens e mínimos por categoria. Para validar/regenerar:

```bat
venv\Scripts\python.exe scripts\validate_library.py
venv\Scripts\python.exe scripts\seed_library.py
```

Regenerar sobrescreve a biblioteca editada; faça backup antes.

## Persistência e pastas

- `data/settings.json`: preferências persistentes.
- `data/cliprush_local.db`: fila, posts, calendário, métricas, checklist, índice de renders e textos por clipe.
- `uploads/`: cópias dos uploads de browser; caminhos locais não são copiados.
- `output/sessoes/`: análises, CSVs, progresso, renders, capas e histórico.
- `models/`, `temp/`, `logs/`: cache/modelos locais, temporários e logs.

Use **Configurações → Preparar backup local** para baixar banco, preferências e biblioteca. Vídeos e gameplays são dados do usuário: guarde cópias separadas. O ZIP de distribuição não inclui banco, `.venv`, gameplays, resultados, modelos ou arquivos pessoais.

## CLI, dependências e validação

```bat
venv\Scripts\python.exe analyzer.py "D:\Gameplays\partida.mp4" --mode quick --debug
venv\Scripts\python.exe renderer.py --analysis-json "output\sessoes\SESSAO\analysis.json" --rank 1 --platform shorts --captions
venv\Scripts\python.exe -m unittest discover -s tests -v
venv\Scripts\python.exe -m pip check
```

O analyzer grava JSON, CSV de auditoria, log e progresso por sessão. Cutscenes podem ser mantidas (padrão), priorizando gameplay ou ignoradas; fala, música e tensão formam o score de cenas, enquanto movimento/cortes não ganham peso de ação sem HUD. Os testes automatizados rodam no Sandbox Linux e cobrem o contrato Ollama simulado, extração real de keyframes de um vídeo sintético, fallback, render e persistência. O notebook Windows ainda deve validar PATH/codec de FFmpeg, modelo Ollama efetivamente instalado e permissões locais.

### Flags adicionais de IA no analyzer

A execução CLI também aceita `--no-vision-ai`, `--ollama-url`, `--vision-model`, `--text-model`, `--vision-timeout` e `--max-vision-candidates`. Exemplo:

```bat
venv\Scripts\python.exe analyzer.py "D:\Gameplays\partida.mp4" --vision-model llava:7b --text-model llama3.2:3b --max-vision-candidates 18
```
