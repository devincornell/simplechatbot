from __future__ import annotations
import typing
import getpass
import os

import langchain_core.tools

import sys
sys.path.append('..')
import simplechatbot
from simplechatbot.devin.openai import OpenAIChatBot
import unittest
from unittest.mock import MagicMock
from simplechatbot.andrew.v1.chatbot import LLMWithHistory, HumanMessage, SystemMessage, AIMessage

class TestLLMWithHistory(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.chatbot = LLMWithHistory(llm=self.mock_llm)

    def test_initialization(self):
        self.assertEqual(self.chatbot.history, [])
        self.assertIsNone(self.chatbot.system_message)
        self.assertIsNone(self.chatbot.tools)
        self.assertFalse(self.chatbot.rag)
        self.assertEqual(self.chatbot.cost, 0)

    def test_send_message_to_llm_with_system_message(self):
        self.chatbot.system_message = "System prompt"
        self.chatbot.send_message_to_llm("Hello")
        self.assertIsInstance(self.chatbot.history[0], SystemMessage)
        self.assertEqual(self.chatbot.history[0].content, "System prompt")
        self.assertIsInstance(self.chatbot.history[1], HumanMessage)
        self.assertEqual(self.chatbot.history[1].content, "Hello")

    def test_send_message_to_llm_without_system_message(self):
        self.chatbot.send_message_to_llm("Hello")
        self.assertIsInstance(self.chatbot.history[0], HumanMessage)
        self.assertEqual(self.chatbot.history[0].content, "Hello")

    def test_trim_history(self):
        self.chatbot.history = [
            SystemMessage("System prompt"),
            HumanMessage("Message 1"),
            AIMessage("Response 1"),
            HumanMessage("Message 2"),
            AIMessage("Response 2"),
            HumanMessage("Message 3"),
        ]
        self.chatbot.send_message_to_llm("Message 4")
        self.assertEqual(len(self.chatbot.history), 5)
        self.assertIsInstance(self.chatbot.history[0], SystemMessage)
        self.assertIsInstance(self.chatbot.history[1], HumanMessage)
        self.assertEqual(self.chatbot.history[1].content, "Message 2")

    def test_tool_calls(self):
        self.chatbot.tools = [MagicMock()]
        self.chatbot.from_list(self.chatbot.tools)
        self.mock_llm.invoke.return_value = AIMessage(
            content="Response",
            response_metadata={'finish_reason': 'tool_calls'},
            tool_calls=[{'name': 'tool1', 'args': {}, 'id': '1'}]
        )
        self.chatbot.tool_dict = {'tool1': MagicMock()}
        self.chatbot.tool_dict['tool1'].invoke.return_value = "Tool response"
        self.chatbot.send_message_to_llm("Hello")
        self.assertEqual(self.chatbot.history[-1].content, "Response")

if __name__ == '__main__':
    unittest.main()
