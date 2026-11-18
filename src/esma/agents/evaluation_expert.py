"""Memory retrieval and evaluation support."""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import time

class EvaluationExpert:
    def __init__(self, memory_file1=None, memory_file2=None, memory_file3=None):
        self.ep_history = []
        self.observation = None
        self.action = None
        self.memory_files = {
            'early': memory_file1,
            'mid': memory_file2,
            'late': memory_file3
        }
        self.current_phase = 'early'

        self.memories = {
            'early': [],
            'mid': [],
            'late': []
        }
        self.embeddings = {
            'early': np.array([]),
            'mid': np.array([]),
            'late': np.array([])
        }

        model_cache_dir = Path(os.getenv('ESMA_MODEL_CACHE', Path.home() / '.cache' / 'esma'))
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)

        self._load_memories()
        self._build_all_vector_indices()

    def get_memory(self, obs, action):
        self.observation = obs
        self.action = action
        inference_result = {
            "observation": self.observation,
            "action": self.action
        }
        self.ep_history.append(inference_result)

    def switch_phase(self, phase_name: str):
        """
        Switch the active memory phase.

        Args:
            phase_name: 'early', 'mid', or 'late'
        """
        if phase_name in self.memories and phase_name != self.current_phase:
            self.current_phase = phase_name
            if len(self.embeddings[phase_name]) == 0 and len(self.memories[phase_name]) > 0:
                self._build_phase_vector_index(phase_name)

    def build_memory_pool(self, filename):
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(self.ep_history, f, ensure_ascii=False)
            f.write('\n')

    def _load_memories(self):
        """Load JSON memory records for every phase."""
        for phase, file_path in self.memory_files.items():
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:

                        data = json.load(f)

                        if isinstance(data, list):

                            self.memories[phase] = [
                                item for item in data
                                if isinstance(item, dict) and 'observation' in item and 'action' in item
                            ]
                        else:
                            self.memories[phase] = []

                except json.JSONDecodeError as e:
                    self.memories[phase] = []
                except Exception as e:
                    self.memories[phase] = []
            else:
                self.memories[phase] = []

    def _build_all_vector_indices(self):
        """Build vector indexes for every phase."""
        for phase in ['early', 'mid', 'late']:
            self._build_phase_vector_index(phase)

    def _build_phase_vector_index(self, phase: str):
        """Build the vector index for one phase."""
        memories = self.memories[phase]
        if not memories:
            print(f"The {phase} memory is empty; skipping vector index construction")
            self.embeddings[phase] = np.array([])
            return

        texts = [json.dumps(exp['observation'], sort_keys=True) for exp in memories]
        embeddings = self.model.encode(texts)

        if isinstance(embeddings, list):
            self.embeddings[phase] = np.array(embeddings)
        else:
            self.embeddings[phase] = embeddings

        if self.embeddings[phase].ndim == 1:
            self.embeddings[phase] = self.embeddings[phase].reshape(1, -1)

    def search_similar(self, phase, current_obs: Dict, threshold: float = 0.6, max_results: int = 5):
        """
        Search for similar states in the active phase memory.
        """

        current_memories = self.memories[phase]
        current_embeddings = self.embeddings[phase]

        size = len(current_memories)

        if size >= 50:
            if len(current_embeddings) == 0:
                self._build_phase_vector_index(phase)
                current_embeddings = self.embeddings[phase]

            query_text = json.dumps(current_obs, sort_keys=True)
            query_vector = self.model.encode([query_text])[0]

            if current_embeddings.ndim == 1:
                embeddings_2d = current_embeddings.reshape(1, -1)
            else:
                embeddings_2d = current_embeddings

            embeddings_norm = embeddings_2d / np.linalg.norm(embeddings_2d, axis=1, keepdims=True)
            query_vector_norm = query_vector / np.linalg.norm(query_vector)

            similarities = np.dot(embeddings_norm, query_vector_norm.T).flatten()

            above_threshold_indices = np.where(similarities >= threshold)[0]

            if len(above_threshold_indices) > 0:
                above_threshold_similarities = similarities[above_threshold_indices]
                sorted_indices = np.argsort(above_threshold_similarities)[::-1]
                sorted_indices = above_threshold_indices[sorted_indices]

                if max_results is not None and len(sorted_indices) > max_results:
                    sorted_indices = sorted_indices[:max_results]
            else:
                sorted_indices = above_threshold_indices

            results = []
            for idx in sorted_indices:
                current_exp = current_memories[idx]
                action = current_exp['action']
                current_observation = current_exp['observation']

                next_observation = None
                if idx + 1 < len(current_memories):
                    next_exp = current_memories[idx + 1]
                    next_observation = next_exp['observation']

                results.append({
                    'similarity': similarities[idx],
                    'current_observation': current_observation,
                    'current_action': action,
                    'next_observation': next_observation,
                    'phase': phase
                })

            return results
        else:
            return []
