"""Project-owned token traversal adapters; the third-party package stays unchanged."""
from dataclasses import dataclass
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.tokenenforcer import TokenEnforcer


class ExactTokenParser(CharacterLevelParser):
    """Traverse the token trie through the original character parser.

    The pinned enforcer's JSON free-text shortcut drops legal tokens containing
    text followed by a closing quote and punctuation (for example `.\",`).
    No tokens are added by hand and no third-party implementation is patched.
    """

    def __init__(self, parser):
        self.parser = parser

    @property
    def config(self):
        return self.parser.config

    @config.setter
    def config(self, value):
        self.parser.config = value

    def add_character(self, character):
        return ExactTokenParser(self.parser.add_character(character))

    def get_allowed_characters(self):
        return self.parser.get_allowed_characters()

    def can_end(self):
        return self.parser.can_end()

    def cache_key(self):
        return self.parser.cache_key()

    def shortcut_key(self):
        return None


@dataclass
class _BoundaryNode:
    tokens: list
    children: dict


def _boundary_trie(node, *, quote_seen=False, after_quote=False, root=True):
    """Keep actual vocabulary paths with an internal quote, excluding leading quotes.

    The library already traverses leading-quote tokens. Its free-text cache
    excludes internal-quote tokens with trailing characters, which this trie
    supplies to the SAME character parser, without declaring any token legal.
    """
    children = {}
    for character, child in node.children.items():
        if root and character == '"':
            continue
        retained = _boundary_trie(child, quote_seen=quote_seen or character == '"',
                                  after_quote=after_quote or quote_seen, root=False)
        if retained is not None:
            children[character] = retained
    tokens = list(node.tokens) if after_quote else []
    return _BoundaryNode(tokens, children) if tokens or children else None


class BoundaryAwareTokenEnforcer(TokenEnforcer):
    def __init__(self, tokenizer_data, parser, boundary_trie):
        super().__init__(tokenizer_data, parser)
        self.boundary_trie = boundary_trie

    def _collect_allowed_tokens(self, parser, tree_node, allowed_tokens, shortcut_key):
        super()._collect_allowed_tokens(parser, tree_node, allowed_tokens, shortcut_key)
        if (isinstance(shortcut_key, tuple) and shortcut_key[0] == 'json_freetext'
                and self.boundary_trie is not None):
            super()._collect_allowed_tokens(parser, self.boundary_trie, allowed_tokens, None)


def boundary_token_trie(tokenizer_data):
    return _boundary_trie(tokenizer_data.tokenizer_tree.root)


def build_boundary_aware_prefix_fn(tokenizer_data, parser, boundary_trie):
    enforcer = BoundaryAwareTokenEnforcer(tokenizer_data, parser, boundary_trie)

    def allowed(batch_id, tokens):
        return enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens

    return allowed
