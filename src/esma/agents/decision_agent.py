"""LLM action-selection client."""

import json
import re
import time

class DecisionAgent:
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.input_tokens = 0
        self.ouput_tokens = 0
        self.inference_time = 0

    def system_prompt(self, n_agents, n_enemies, agent_tag_list, enemy_tag_list) -> str:
        return f"""You are a StarCraft II multi-agent tactical controller.
        Select actions for {n_agents} allied units against {n_enemies} enemy units.
        Movement: 102 north, 103 south, 104 east, 105 west. Attack: 106 through {105 + n_enemies}.
        Return only JSON: {{"actions": [action_id]}} with exactly {n_agents} entries."""

    def user_prompt(self, obs, n_agents, summary):
        return f"Battle state: {obs}\nHistory summary: {summary}\nReturn {n_agents} actions."

    async def _get_actions2(self, system_prompt, user_prompt):
        """Call the LLM asynchronously and parse tuple-style responses."""
        from openai import OpenAI

        client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

        try:
            start_time = time.time()
            first_token_time = None
            full_response = ""
            received_first_token = False

            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                reasoning_effort="minimal"
            )

            for chunk in stream:

                if not received_first_token and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                    ttf = first_token_time - start_time
                    received_first_token = True
                    self.inference_time = ttf
                    print(f"LLM response time: {ttf:.3f} seconds")

                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            completion = {
                "choices": [{"message": {"content": full_response}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

            response_text = completion["choices"][0]["message"]["content"]
            print(f"Raw LLM response:\n{response_text}")

            import re
            response_text = re.sub(r"//.*", "", response_text)
            response_text = re.sub(r"```json\n?", "", response_text)
            response_text = re.sub(r"```", "", response_text)
            response_text = response_text.strip()

            try:

                actions_data = json.loads(response_text)

                if not isinstance(actions_data, list):
                    print(f"Response is not a list: {type(actions_data)}")
                    return [], completion

                validated_actions = []
                for i, item in enumerate(actions_data):
                    if not isinstance(item, dict):
                        print(f"Item {i} is not a dictionary: {item}")
                        continue

                    if "ut" not in item or "ai" not in item:
                        print(f"Item {i} is missing required fields: {item}")
                        continue

                    try:
                        unit_tag = int(item["ut"])
                        action_id = int(item["ai"])
                    except (ValueError, TypeError):
                        print(f"Item {i} has invalid data types: {item}")
                        continue

                    action_info = {
                        "ut": unit_tag,
                        "ai": action_id
                    }

                    if "move_distance" in item:
                        try:
                            move_distance = float(item["move_distance"])

                            if move_distance < 0:
                                move_distance = 0
                            elif move_distance > 10:
                                move_distance = 10
                            action_info["move_distance"] = move_distance
                            print(f"Parsed movement distance for item {i}: {move_distance}")
                        except (ValueError, TypeError):
                            print(f"Invalid move_distance for item {i}: {item.get('move_distance')}")

                    if action_id >= 106:
                        if "tt" in item:
                            try:
                                action_info["tt"] = int(item["tt"])
                            except (ValueError, TypeError):

                                if "target_position" in item:
                                    pos_str = item["target_position"]

                                    match = re.search(r'Point2\(\(([\d.]+),\s*([\d.]+)\)\)', pos_str)
                                    if match:
                                        action_info["target_position"] = (float(match.group(1)), float(match.group(2)))
                        elif "target_position" in item:
                            pos_str = item["target_position"]

                            match = re.search(r'Point2\(\(([\d.]+),\s*([\d.]+)\)\)', pos_str)
                            if match:
                                action_info["target_position"] = (float(match.group(1)), float(match.group(2)))
                            else:

                                match = re.search(r'\(([\d.]+),\s*([\d.]+)\)', pos_str)
                                if match:
                                    action_info["target_position"] = (float(match.group(1)), float(match.group(2)))
                                else:
                                    print(f"Unable to parse target_position for item {i}: {pos_str}")

                    validated_actions.append(action_info)

                return validated_actions, completion

            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Raw text: {response_text}")

                try:

                    start_idx = response_text.find('[')
                    end_idx = response_text.rfind(']') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = response_text[start_idx:end_idx]
                        actions_data = json.loads(json_str)
                        print("Extracted a JSON array from the response")
                        return actions_data, completion
                except Exception as inner_e:
                    print(f"JSON extraction also failed: {inner_e}")

            return [], completion

        except Exception as e:
            print(f"LLM API call failed: {e}")
            return [], None

    async def _get_actions(self, system_prompt, user_prompt):
        """Call the LLM asynchronously through the current API."""
        from openai import OpenAI
        import json
        import re

        client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

        try:
            print("Calling the LLM API...")

            start_time = time.time()
            full_response = ""
            received_first_token = False

            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                reasoning_effort="minimal"
            )

            for chunk in stream:

                if not received_first_token and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                    ttf = first_token_time - start_time
                    received_first_token = True
                    self.inference_time = ttf
                    print(f"LLM response time: {ttf:.3f} seconds")

                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            completion = {
                "choices": [{"message": {"content": full_response}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

            response_text = completion["choices"][0]["message"]["content"]
            print(f"Raw LLM response:\n{response_text}")

            if response_text is None:
                print("The LLM returned no content")
                return [], completion

            cleaned_text = response_text.strip()

            print(f"Cleaned response (first 500 characters): {cleaned_text[:500]}...")

            number_actions = self._parse_number_list(cleaned_text)
            if number_actions:

                return number_actions, completion

            print("Unable to parse the response")
            return [], completion

        except Exception as e:
            print(f"LLM API call failed: {e}")
            import traceback
            traceback.print_exc()
            return [], None

    def _parse_json_actions(self, text):
        """Parse a JSON-formatted action list."""
        import json
        import re

        try:

            if text.startswith('[') and text.endswith(']'):
                try:
                    actions = json.loads(text)
                    if isinstance(actions, list):
                        return actions
                except:
                    pass

            json_pattern = r'\[(?:{[^}]+},?\s*)+\]'
            matches = re.findall(json_pattern, text, re.DOTALL)

            for match in matches:
                try:
                    actions = json.loads(match)
                    if isinstance(actions, list) and len(actions) > 0:
                        return actions
                except json.JSONDecodeError:
                    continue

            object_pattern = r'\{[^{}]*\}'
            matches = re.findall(object_pattern, text, re.DOTALL)

            if matches:
                actions = []
                for match in matches:
                    try:
                        action = json.loads(match)
                        if isinstance(action, dict):
                            actions.append(action)
                    except:
                        continue

                if actions:
                    return actions

            return None

        except Exception as e:
            print(f"JSON parsing error: {e}")
            return None

    def _parse_number_list(self, text):
        """Parse a plain numeric list."""
        import re

        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)

        if numbers:
            try:

                numbers = [int(n) for n in numbers[:5]]
                return numbers
            except:
                return None

        return None

    def _is_number_list_format(self, text):
        """Check whether text is a plain numeric list."""

        cleaned = re.sub(r'^\[|\]$', '', text.strip())

        return bool(re.match(r'^[\d,\s]+$', cleaned))

    def _parse_number_string(self, text):
        """Parse a numeric string into a list of integers."""

        cleaned = re.sub(r'^\[|\]$', '', text.strip())
        numbers = []

        for num_str in cleaned.split(','):
            num_str = num_str.strip()
            if num_str.isdigit():
                numbers.append(int(num_str))

        return numbers

    def _extract_numbers_from_text(self, text):
        """Extract all integers from text."""
        numbers = re.findall(r'\b\d+\b', text)
        return [int(num) for num in numbers if num.isdigit()]
