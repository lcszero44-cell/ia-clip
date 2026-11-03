from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import atomic_write_json
from library import build_library_document

CATALOG = {
    "ganchos": [
        "Espera até você ver o que acontece no final.",
        "Eu achei que essa parte seria tranquila…",
        "O jogo mudou tudo em dois segundos.",
        "Repara no canto da tela antes do susto.",
        "Esse detalhe passou despercebido na primeira vez.",
        "Eu não estava preparado para essa virada.",
        "A decisão mais arriscada da partida foi essa.",
        "Parece impossível escapar, mas olha só.",
        "O silêncio antes desse momento é assustador.",
        "Tudo estava sob controle… até deixar de estar.",
        "Você faria a mesma escolha aqui?",
        "O plano era simples. O resultado, nem tanto.",
        "Assiste até o último segundo e me diz se percebeu.",
        "Uma bala. Um erro. Uma chance.",
        "Esse foi o momento em que a run quase acabou.",
        "Eu ouvi o barulho, mas não vi de onde veio.",
        "Presta atenção na reação do jogo depois disso.",
        "A melhor parte não foi o tiro — foi o que veio depois.",
        "Por um instante, eu achei que tinha perdido tudo.",
        "A câmera virou e eu já sabia que vinha problema.",
        "Este corredor nunca pareceu tão longo.",
        "O inventário estava cheio. O perigo, também.",
        "Quando a munição acaba, cada segundo conta.",
        "Esse encontro foi muito mais tenso do que parecia.",
        "Eu quase passei direto por esse detalhe.",
        "A sorte ajudou, mas a mira teve que acompanhar.",
        "Um segundo de calma antes do caos completo.",
        "O jogo deu uma pista — eu só entendi tarde demais.",
        "Essa cena merece ser vista sem piscar.",
        "O que você faria se estivesse no meu lugar?",
    ],
    "titulos_acao": [
        "{evento} no momento perfeito", "A jogada que salvou a run", "Como eu escapei por um fio",
        "Essa mira não podia falhar", "O combate mais tenso da partida", "Um erro e tudo acabava",
        "A virada aconteceu em segundos", "O susto que eu não esperava", "A estratégia funcionou no limite",
        "Quando o plano dá certo por pouco", "A esquiva que mudou a luta", "O momento mais caótico da gameplay",
        "Essa sequência foi pura adrenalina", "Sem munição e sem espaço para errar", "O inimigo não viu isso chegando",
        "A reação mais rápida da run", "Uma chance. Uma decisão.", "Eu quase perdi esse combate",
        "A luta virou no último instante", "O golpe que abriu caminho", "O detalhe que decidiu a partida",
        "Uma jogada limpa no meio do caos", "Essa defesa foi no limite", "O confronto que virou história",
        "A run ficou por um fio", "O melhor timing que consegui", "Calma, mira e reação rápida",
        "O jogo apertou — eu respondi", "A sequência que eu queria repetir", "Esse combo saiu sem querer",
        "Não tinha espaço para recuar", "A mira encontrou o alvo", "A esquiva foi por centímetros",
        "Como sobreviver a esse ataque", "O plano B salvou tudo", "O clímax dessa batalha chegou",
        "Essa foi a decisão certa?", "O combate não acabou quando parecia", "A jogada que mudou o ritmo",
    ],
    "titulos_cena": [
        "A cena que mudou o rumo da história", "Você percebeu esse detalhe?", "O diálogo que deixou tudo mais tenso",
        "Essa revelação muda a perspectiva", "Um momento importante da história", "A pausa antes da próxima surpresa",
        "O que essa cena está tentando nos dizer?", "A história ficou ainda mais sombria", "Uma conversa que merece atenção",
        "A pista estava bem na nossa frente", "Essa passagem explica muita coisa", "O tom da história mudou aqui",
        "Uma escolha difícil para a protagonista", "O silêncio diz mais que a fala", "Essa revelação me pegou desprevenido",
        "Um detalhe para lembrar nos próximos capítulos", "Você viu o que apareceu ao fundo?", "A história prepara algo maior",
        "Essa cena ficou na minha cabeça", "O momento de respirar antes do caos",
    ],
    "descricoes": [
        "Gameplay sem cortes na tela principal, com foco no momento e no contexto da partida.",
        "Um trecho da run que merece ser revisto. Qual foi a sua parte favorita?",
        "Clipes de gameplay e momentos da história em português. Sem webcam, só o jogo.",
        "A tensão foi crescendo até chegar neste instante. Você teria feito diferente?",
        "Mais um recorte da minha campanha. Deixe sua leitura desse momento nos comentários.",
        "Uma sequência da gameplay que quase saiu do controle. Veja até o final.",
        "Registro da partida com HUD preservada e contexto do que aconteceu.",
        "Este momento mistura estratégia, tensão e uma decisão rápida.",
        "Trecho da campanha para quem curte ação e survival horror.",
        "A história avançou e deixou novas perguntas. O que você acha que vem agora?",
        "Um encontro intenso da run. Sem narração inventada: apenas o áudio original do jogo.",
        "Gameplay local editada para vídeo vertical, mantendo a tela do jogo completa.",
        "A partida estava tranquila até esta situação aparecer.",
        "Se você também joga, conta como resolveria este encontro.",
        "Um momento curto da campanha, preservando o contexto e o áudio do jogo.",
        "Hoje a estratégia foi sobreviver primeiro e entender depois.",
        "Recorte de gameplay para rever a jogada e aprender com o que aconteceu.",
        "A cena traz pistas que podem passar batido. Reparou em alguma?",
        "O jogo criou a tensão; eu só tentei acompanhar.",
        "Partida capturada no PS4 e preparada localmente para Shorts, Reels e TikTok.",
        "Um clipe da campanha sem webcam e sem esconder a HUD.",
        "O melhor momento da sequência aparece no gancho. Veja o contexto completo.",
        "Mais um capítulo da gameplay. O que você faria com esses recursos?",
        "A tensão veio da trilha, do diálogo e do que estava acontecendo na tela.",
        "Um recorte de jogo para conversar sobre escolhas, detalhes e estratégia.",
        "Esta sequência ficou ainda melhor quando revi a gravação.",
        "O momento parece simples, mas muda o ritmo da partida.",
        "Gameplay narrada pelo próprio jogo, sem webcam e com legendas opcionais.",
        "Assista e diga se você percebeu a mudança antes de ela acontecer.",
        "Compartilhe com alguém que reconhece este capítulo da campanha.",
    ],
    "ctas": [
        "Comenta: você teria feito diferente?", "Qual foi o detalhe que você notou primeiro?",
        "Salva para rever a estratégia depois.", "Marca alguém que também jogaria essa parte.",
        "Qual nota você dá para essa sequência, de 0 a 10?", "Você já passou por esse encontro? Conta como foi.",
        "Quer ver mais trechos dessa campanha? Deixa o próximo jogo nos comentários.",
        "Escolhe: agir rápido ou esperar mais uma pista?", "Comenta qual seria seu plano B.",
        "Se curtiu a gameplay, acompanha os próximos capítulos.", "Qual momento te deixou mais tenso?",
        "O que você teria feito com a munição que sobrou?", "Reparou na pista escondida? Escreve onde ela aparece.",
        "Marca quem sempre percebe os detalhes antes de todo mundo.", "Quer uma análise dessa jogada? Peça nos comentários.",
        "Escolhe o próximo desafio da run.", "Compartilha com seu parceiro de survival horror.",
        "Qual parte merece um replay completo?", "Conta sua teoria sobre o que acabou de acontecer.",
        "Você prefere cenas da história ou combate?", "Se faria diferente, explica seu caminho.",
        "Qual jogo merece o próximo vídeo?", "Segue para acompanhar o restante da campanha.",
        "Deixa uma dica para a próxima tentativa.", "A gente continua a campanha no próximo clipe.",
    ],
    "comentarios_fixados": [
        "Qual detalhe você percebeu antes de mim?", "Você teria escolhido a mesma estratégia?",
        "Querem a continuação desta parte?", "Qual nota esse momento merece?",
        "Sem spoiler: o que vocês acham que acontece depois?", "Você joga com qual configuração de dificuldade?",
        "Qual foi a melhor decisão dessa sequência?", "Qual arma você levaria para essa parte?",
        "Quem também ficou tenso nessa cena?", "Qual detalhe do cenário chamou sua atenção?",
        "Esse momento foi sorte ou planejamento?", "Vocês preferem gameplay com legenda ou sem?",
        "Qual chefe foi mais difícil para você?", "Que jogo eu deveria jogar depois dessa campanha?",
        "Qual foi seu primeiro pensamento quando isso aconteceu?", "Dica sem spoiler para quem ainda não chegou aqui?",
        "Qual rota você usaria nesta situação?", "Querem um vídeo com a estratégia completa?",
        "O que eu deveria ter feito diferente?", "Comenta sua teoria sem estragar a surpresa para ninguém.",
    ],
    "reacoes_acao": [
        "HEADSHOT!", "NO LIMITE!", "QUE ESQUIVA!", "COMBO INSANO!", "ERA A ÚLTIMA BALA!",
        "QUASE DEU RUIM!", "MIRA PERFEITA!", "VIRADA ABSURDA!", "DEFESA NO TEMPO CERTO!",
        "AÍ NÃO!", "ALVO ENCONTRADO!", "SOBREVIVI POR UM FIO!", "QUE TIMING!",
        "PLANO B ATIVADO!", "BOSS NO CHÃO!", "ESCAPEI!", "SEM ERRO!", "AGORA VAI!",
        "UM SEGUNDO!", "QUE PRESSÃO!",
    ],
    "reacoes_cena": [
        "VOCÊ VIU ISSO?", "A PISTA ESTAVA AÍ", "ESSA REVELAÇÃO…", "OLHA O DETALHE",
        "AGORA TUDO FAZ SENTIDO", "UM MOMENTO IMPORTANTE", "QUE CLIMA…", "A HISTÓRIA VIROU",
        "PRESTA ATENÇÃO", "O SILÊNCIO DIZ TUDO", "ISSO MUDA TUDO", "VAMOS LEMBRAR DISSO",
        "QUE CENA", "UMA NOVA PISTA", "E AGORA?",
    ],
    "conjuntos_hashtags": [
        "#gameplay #gaming #ps4 #shorts", "#survivalhorror #gameplayptbr #shorts #ps4",
        "#residentevil #re4remake #gameplay #brasil", "#gamerbr #jogos #gameplay #verticalvideo",
        "#youtubeShorts #gameplay #survivalhorror #semwebcam", "#tiktokgaming #gamingbr #gameplay #ps4",
        "#reelsgaming #gameplaybr #jogos #survivalhorror", "#playstation #ps4 #gaming #gameplay",
        "#terror #horror #gameplay #brasil", "#dicasdegames #estrategia #gameplay #gamerbr",
        "#cenasdegames #historiadejogo #gaming #shorts", "#bossfight #combate #gameplay #gamer",
        "#momentosdegames #clips #gamingbr #shorts", "#streamerbrasil #gameplay #ps4 #gamer",
        "#jogosdeacao #survival #gameplayptbr #reels", "#playstationbrasil #gameplay #shorts #jogos",
        "#semspoiler #gameplay #historia #gamerbr", "#capturadegame #ps4 #clips #gaming",
        "#re4 #residentevil4 #survivalhorror #gamerbr", "#aventura #campanha #gameplay #shorts",
    ],
    "roteiros_narracao": [
        "A ideia era manter distância, observar o padrão e agir quando surgisse uma abertura.",
        "Eu entrei com poucos recursos, então cada decisão precisava fazer sentido.",
        "A trilha começou a subir e ficou claro que a próxima área não seria tranquila.",
        "Antes de avançar, vale olhar o cenário: havia uma pista que quase passou despercebida.",
        "A estratégia mudou quando percebi que a rota de volta estava bloqueada.",
        "Eu precisava escolher entre gastar munição agora ou guardar para depois.",
        "A cena parece calma, mas a informação que ela entrega é importante para a história.",
        "O plano foi simples: esperar o ataque, esquivar e usar a abertura.",
        "Quando o inimigo mudou o padrão, eu tive que improvisar.",
        "A tensão não veio só do combate, mas do pouco espaço para escapar.",
        "Eu achei que tinha entendido a mecânica, até o jogo mudar as regras.",
        "Nesse trecho, prestar atenção no áudio foi tão importante quanto olhar a HUD.",
        "A munição estava curta, então tentei priorizar os alvos mais próximos.",
        "O detalhe no fundo ajuda a explicar por que essa cena parece familiar.",
        "Eu recuei para ganhar espaço e consegui recuperar o controle da luta.",
        "A decisão foi arriscada, mas abriu uma rota melhor para continuar.",
        "O mais importante aqui é reconhecer o momento certo para não atacar.",
        "A história deixa uma pergunta no ar, e eu quero ver como isso vai continuar.",
        "Depois de rever a gravação, percebi que havia outra saída disponível.",
        "Essa foi a última tentativa antes de mudar completamente a estratégia.",
    ],
}


def main() -> int:
    document = build_library_document(CATALOG)
    output = ROOT / "biblioteca" / "biblioteca_editorial.json"
    atomic_write_json(output, document)
    print(f"Biblioteca gerada: {output}")
    print(f"Itens: {document['item_count']}")
    for category, count in sorted(document["category_counts"].items()):
        print(f"{category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
