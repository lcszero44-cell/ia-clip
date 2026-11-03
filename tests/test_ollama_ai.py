"""Testes da integração Ollama sem iniciar nem consultar um servidor real."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ollama_ai


def _resposta_json(payload: object) -> MagicMock:
    """Cria uma resposta HTTP falsa compatível com o gerenciador de contexto."""

    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return response


class OllamaAITests(unittest.TestCase):
    """Verifica o contrato seguro da integração local."""

    def test_check_status_consulta_tags_e_identifica_modelo(self):
        """O status extrai somente nomes de modelos da resposta de tags."""

        payload = {"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:3b"}]}
        with patch("ollama_ai.urllib.request.urlopen", return_value=_resposta_json(payload)) as abrir:
            status = ollama_ai.check_status()

        self.assertTrue(status["ok"])
        self.assertTrue(status["model_installed"])
        self.assertEqual(status["models"], ["llama3.2:3b", "qwen2.5:3b"])
        request = abrir.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/tags")
        self.assertEqual(request.get_method(), "GET")

    def test_modelo_ausente_e_reportado_sem_lancar(self):
        """O Ollama disponível sem o modelo pedido não é tratado como instalado."""

        payload = {"models": [{"name": "outro-modelo:1b"}]}
        with patch("ollama_ai.urllib.request.urlopen", return_value=_resposta_json(payload)):
            status = ollama_ai.check_status(model="llama3.2:3b")
        self.assertTrue(status["ok"])
        self.assertFalse(status["model_installed"])
        self.assertIn("não está instalado", status["message"])

        # Um erro de modelo no endpoint de geração também deve ser opcional.
        erro = HTTPError(
            "http://localhost:11434/api/chat", 404, "modelo ausente", {}, None
        )
        with patch("ollama_ai.urllib.request.urlopen", side_effect=erro):
            self.assertIsNone(
                ollama_ai.generate_editorial_copy({"jogo": "Meu jogo"})
            )

    def test_generate_envia_chat_json_conciso_e_apenas_metadados_permitidos(self):
        """A geração usa JSON, contexto pequeno e não envia caminhos locais."""

        editorial = {
            "titles": ["Título um", "Título dois", "Título três"],
            "hook": "Olha esse momento!",
            "description": "Um momento marcante da partida.",
            "cta": "Comente o que você faria.",
            "hashtags": "#gaming #shorts",
            "pinned_comment": "Qual foi sua reação?",
        }
        resposta = {"message": {"role": "assistant", "content": json.dumps(editorial)}}
        contexto = {
            "jogo": "Meu jogo",
            "plataforma": "shorts",
            "evento_seguro": "escape",
            "score": 88.5,
            "duracao": 17,
            "cena_fala": "A fala confirma o perigo.",
            "resumo": "Escape no último segundo.",
            "video_path": "/home/ubuntu/cliprush/segredo.mp4",
            "classification": "não enviar",
            "ranking": 1,
        }
        with patch("ollama_ai.urllib.request.urlopen", return_value=_resposta_json(resposta)) as abrir:
            resultado = ollama_ai.generate_editorial_copy(contexto)

        self.assertEqual(resultado, editorial)
        request = abrir.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "llama3.2:3b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], "json")
        self.assertIn(body["options"]["num_ctx"], (4096, 8192))
        self.assertEqual(body["options"]["num_gpu"], 0)
        self.assertEqual(body["options"]["num_thread"], 4)
        self.assertEqual(body["keep_alive"], "2m")
        serialized = request.data.decode("utf-8")
        self.assertNotIn("video_path", serialized)
        self.assertNotIn("/home/ubuntu/cliprush/segredo.mp4", serialized)
        self.assertNotIn("classification", serialized)
        self.assertNotIn('"ranking":', serialized)
        self.assertIn("PT-BR", body["messages"][0]["content"])
        self.assertIn("Nunca invente", body["messages"][0]["content"])

    def test_falha_de_conexao_retorna_none(self):
        """Falha de conexão no endpoint de geração não vaza exceção."""

        with patch(
            "ollama_ai.urllib.request.urlopen",
            side_effect=URLError("Ollama desligado"),
        ):
            resultado = ollama_ai.generate_editorial_copy({"jogo": "Meu jogo"})
        self.assertIsNone(resultado)

    def test_url_remota_e_rejeitada_sem_chamada(self):
        """URLs externas são rejeitadas antes de qualquer envio de dados."""

        with patch("ollama_ai.urllib.request.urlopen") as abrir:
            status = ollama_ai.check_status(base_url="http://example.com:11434")
            resultado = ollama_ai.generate_editorial_copy(
                {"jogo": "Meu jogo"}, base_url="http://8.8.8.8:11434"
            )

        self.assertFalse(status["ok"])
        self.assertFalse(status["model_installed"])
        self.assertIsNone(resultado)
        abrir.assert_not_called()

    def test_saida_parcial_ou_invalida_retorna_none(self):
        """JSON parcial e JSON com quantidade errada de títulos são recusados."""

        respostas = [
            {"message": {"content": json.dumps({"titles": ["só um"]})}},
            {
                "message": {
                    "content": json.dumps(
                        {
                            "titles": ["um", "dois", "três"],
                            "hook": "",
                            "description": "Descrição",
                            "cta": "CTA",
                            "hashtags": "#jogo",
                            "pinned_comment": "Comentário",
                        }
                    )
                }
            },
        ]
        for resposta in respostas:
            with self.subTest(resposta=resposta), patch(
                "ollama_ai.urllib.request.urlopen",
                return_value=_resposta_json(resposta),
            ):
                self.assertIsNone(
                    ollama_ai.generate_editorial_copy({"jogo": "Meu jogo"})
                )

    def test_json_invalido_do_assistente_retorna_none(self):
        """Conteúdo que não é JSON estrito não é aceito como copy."""

        resposta = {"message": {"content": "```json\n{\"titles\": []}\n```"}}
        with patch(
            "ollama_ai.urllib.request.urlopen", return_value=_resposta_json(resposta)
        ):
            self.assertIsNone(
                ollama_ai.generate_editorial_copy({"jogo": "Meu jogo"})
            )


if __name__ == "__main__":
    unittest.main()
