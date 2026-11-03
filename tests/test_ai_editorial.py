from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


AI_COPY = {
    "titles": ["Escape no último segundo", "A partida virou em um instante", "Você teria arriscado?"],
    "hook": "Espere pelo final deste momento.",
    "description": "Um trecho intenso da gameplay, com a tela do jogo preservada.",
    "cta": "O que você faria nessa situação?",
    "hashtags": "#jogos #gameplay",
    "pinned_comment": "Qual foi sua reação?",
}


class EditorialAITests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "ollama_enabled": True,
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "llama3.2:3b",
            "ollama_timeout": 25,
        }

    def test_ollama_texto_valido_mescla_com_hashtags_offline(self):
        with patch("app.ollama_ai.generate_editorial_copy", return_value=AI_COPY):
            kit, used = app._generate_editorial_kit(
                {"event_type": "impacto", "score": 88, "duration_seconds": 30},
                "Meu jogo", "shorts", {"items": []}, self.settings,
            )
        self.assertTrue(used)
        self.assertEqual(kit["titles"], AI_COPY["titles"])
        self.assertIn("#shorts", kit["hashtags"])
        self.assertIn("#gameplay", kit["hashtags"])
        self.assertEqual(kit["reaction_text"], "QUE MOMENTO!")

    def test_falha_ollama_usa_kit_offline_sem_excecao(self):
        with patch("app.ollama_ai.generate_editorial_copy", return_value=None):
            kit, used = app._generate_editorial_kit(
                {"event_type": "impacto", "score": 75}, "", "shorts", {"items": []}, self.settings,
            )
        self.assertFalse(used)
        self.assertTrue(kit["titles"])
        self.assertEqual(kit["event_type"], "impacto")

    def test_texto_que_inventa_motosserra_e_descartado(self):
        invented = {**AI_COPY, "titles": ["A motosserra apareceu do nada", "Título seguro", "Mais um título"]}
        with patch("app.ollama_ai.generate_editorial_copy", return_value=invented):
            kit, used = app._generate_editorial_kit(
                {"event_type": "motosserra", "confidence": 0.58,
                 "classification_reason": "grave sustentado/áspero (heurística)"},
                "Meu jogo", "shorts", {"items": []}, self.settings,
            )
        self.assertFalse(used)
        self.assertNotIn("motosserra", " ".join(kit["titles"]).lower())
        self.assertEqual(kit["event_type"], "impacto")

    def test_ia_desativada_nao_consulta_ollama(self):
        settings = {**self.settings, "ollama_enabled": False}
        with patch("app.ollama_ai.generate_editorial_copy") as generate:
            kit, used = app._generate_editorial_kit(
                {"event_type": "impacto"}, "", "shorts", {"items": []}, settings,
            )
        self.assertFalse(used)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
