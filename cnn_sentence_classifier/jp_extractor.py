import logging
from abc import abstractmethod
import MeCab
import unicodedata

logger = logging.getLogger(__name__)
INDEX_ORIGINAL_FORM = 6
INDEX_WORD_CLASS = 0


class WordExtractor(object):
    """
    """
    @abstractmethod
    def __call__(self, text):
        pass

    def extract_words(self, text):
        raise NotImplementedError('extract_words must be implemented')


class JapaneseMecabWordExtractor(WordExtractor):
    """ Japanese Mecab Word Extractor
            word class)
                http://www.unixuser.org/~euske/doc/postag/
    """
    def __init__(self,
                 split_mode="unigram",
                 use_noun=True,
                 use_verb=True,
                 use_adjective=True,
                 use_numerical=True,
                 dict_ignore_type=None,
                 use_all=False,
                 tagger_option=""):
        self.split_mode = split_mode
        # unigram, ngram, ngram_and_unigram or ngram_or_unigram
        self.use_noun = use_noun
        self.use_verb = use_verb
        self.use_adjective = use_adjective
        self.use_numerical = use_numerical
        self.tagger = MeCab.Tagger(tagger_option)
        self.dict_ignore_type = dict_ignore_type
        self.use_all = use_all

        # これがないとmecabのバグにかかる
        self.tagger.parse("")

    def __call__(self, text):
        return self.extract_words(text)

    def extract_words(self, text):
        """ tokenize
        Args:
            text:
            mode:
        Returns:
            list words
        Raises:
        """
        # if isinstance(text, unicode):
        #     norm_text = unicodedata.normalize("NFKC", text)
        # else:
        #     norm_text = unicodedata.normalize("NFKC", text.decode("utf-8"))
        norm_text = unicodedata.normalize("NFKC", text)

        if self.split_mode == "unigram":
            unigrams = [
                token
                for token in self.jatokenize(norm_text)
                if not token.startswith('_')
            ]
            ngrams = []
        elif self.split_mode == "ngram":
            unigrams = []
            ngrams = [
                token
                for token in self.jatokenize_ngram(norm_text, "ngram_only")
                if not token.startswith('_')
            ]
        elif self.split_mode == "ngram_or_unigram":
            unigrams = []
            ngrams = [
                token
                for token in self.jatokenize_ngram(norm_text, "ngram_or_unigram")
                if not token.startswith('_')
            ]
        else:
            unigrams = [
                token
                for token in self.jatokenize(norm_text)
                if not token.startswith('_')
            ]
            ngrams = [
                token
                for token in self.jatokenize_ngram(norm_text, "ngram_only")
                if not token.startswith('_')
            ]

        return unigrams + ngrams

    def jatokenize(self, text):
        """ jatokenize
        Args:
            text: text
        Returns:
        Yield:
            words
        Raises:
        """
        # if isinstance(text, unicode):
        #     encoded_text = text.encode('utf-8')
        # else:
        #     encoded_text = text
        encoded_text = text
        node = self.tagger.parseToNode(encoded_text).next
        while node:
            if self.check_usable(node):
                yield self.get_original(node)

            ppnode, pnode, node = self.forward(None, None, node, 1)

    def check_usable(self, node):
        """ check usable
        Args:
            node: mecab node:
        Return:
            boolean
        """
        if node.surface == "":
            return False
        if self.use_all:
            return True

        if self.check_class(node, "名詞") and self.use_noun:
            if self.rulebase_stopwords(node):
                return False
            elif self.check_numerical(node):
                if self.use_numerical:
                    return True
            else:
                return True

        elif self.check_class(node, "動詞") and self.use_verb:
            if self.rulebase_stopwords(node):
                return False
            else:
                return True

        elif self.check_class(node, "形容詞") and self.use_adjective:
            if self.rulebase_stopwords(node):
                return False
            else:
                return True
        return False

    def jatokenize_ngram(self, text, mode="ngram_only"):
        """ jatokenize_ngram
        Args:
            text: text
            mode: extraction mode
        Returns:
        Yield:
            words
        Raises:
        """
        # if isinstance(text, unicode):
        #     encoded_text = text.encode('utf-8')
        # else:
        #     encoded_text = text
        encoded_text = text
        node = self.tagger.parseToNode(encoded_text).next
        pnode = None
        ppnode = None
        while node:
            # ngram
            if ppnode is not None:
                # ＊の＊
                if(self.check_class(ppnode, "名詞") and
                   pnode.surface == "の" and
                   self.check_class(node, "名詞")):
                    yield (self.get_original(ppnode, True) +
                           self.get_original(pnode, True) +
                           self.get_original(node, True))
                    ppnode, pnode, node = self.forward(ppnode, pnode, node, 2)
                # 接頭辞＋名詞＋接尾辞
                elif(self.check_class(ppnode, "接頭辞") and
                     self.check_class(pnode, "名詞") and
                     self.check_class(node, "接尾辞")):
                    yield (self.get_original(ppnode, True) +
                           self.get_original(pnode, True) +
                           self.get_original(node, True))
                    ppnode, pnode, node = self.forward(ppnode, pnode, node, 2)
                # 名詞＋名詞＋名詞
                elif(self.check_class(ppnode, "名詞") and
                     self.check_class(pnode, "名詞") and
                     self.check_class(node, "名詞")):
                    yield (self.get_original(ppnode, True) +
                           self.get_original(pnode, True) +
                           self.get_original(node, True))
                    # ppnode, pnode, node = self.forward(ppnode, pnode, node, 2)
                # 名詞＋名詞
                elif(self.check_class(ppnode, "名詞") and
                     self.check_class(pnode, "名詞")):
                    yield self.get_original(ppnode, True) + self.get_original(pnode, True)
                    # ppnode, pnode, node = self.forward(ppnode, pnode, node, 1)
                # 形容詞＋名詞
                elif(self.check_class(ppnode, "形容詞") and
                     self.check_class(pnode, "名詞")):
                    yield self.get_original(ppnode, True) + self.get_original(pnode, True)
                    ppnode, pnode, node = self.forward(ppnode, pnode, node, 1)
                # 接頭辞＋名詞
                elif(self.check_class(ppnode, "接頭辞") and
                     self.check_class(pnode, "名詞")):
                    yield self.get_original(ppnode, True) + self.get_original(pnode, True)
                    ppnode, pnode, node = self.forward(ppnode, pnode, node, 1)
                # 名詞＋接尾辞
                elif(self.check_class(ppnode, "名詞") and
                     self.check_class(pnode, "接尾辞")):
                    yield self.get_original(ppnode, True) + self.get_original(pnode, True)
                    ppnode, pnode, node = self.forward(ppnode, pnode, node, 1)

                elif mode == "ngram_or_unigram":
                    if self.check_class(ppnode, "名詞") and self.use_noun:
                        yield self.get_original(ppnode)

                    if self.check_class(ppnode, "動詞") and self.use_verb:
                        # yield ppnode.feature.split(',')[10]
                        yield self.get_original(ppnode)

            ppnode, pnode, node = self.forward(ppnode, pnode, node, 1)

    @staticmethod
    def forward(ppnode, pnode, node, n):
        """ forward
        Args:
            ppnode: pre pre node of mecab
            pnode: pre node of mecab
            node: node of mecab
            n: step to next
        Returns:
            ppnode, pnode, node
        Raises:
        """
        for _ in range(n):
            ppnode = pnode
            pnode = node
            if node is not None:
                node = node.next
        return ppnode, pnode, node

    def rulebase_stopwords(self, node):
        """ rule base stop words
        Args:
            node: mecab node
        Returns:
            Boolean
        """
        # target_char = node.surface.decode("utf-8")
        target_char = node.surface
        if len(target_char) == 1:
            try:
                char_type = unicodedata.name(target_char)
            except:
                return False
            if "HIRAGANA" in char_type:
                return True
            elif "KATAKANA" in char_type:
                return True
            elif "LATIN" in char_type:
                return True
        return False

    def check_class(self, node, word_class="名詞"):
        """ chcek class
        Args:
            node: mecab node
            word_class: parts of speech
        Returns:
            Boolean
        Raises:
        """
        if(node.feature.split(',')[INDEX_WORD_CLASS] == word_class):
            if self.dict_ignore_type is None:
                return True

            types_ = node.feature.split(",")[
                INDEX_WORD_CLASS:INDEX_WORD_CLASS+4
            ]
            depth = self.dict_ignore_type
            for type_ in types_:
                class_depth = depth.get(type_, None)
                if class_depth is None:
                    return True
                elif class_depth == "ignore":
                    return False
                elif isinstance(class_depth, dict) is True:
                    depth = class_depth
                else:
                    return True
        else:
            return False

    def check_numerical(self, node):
        """ check numerical
        Args:
            node: mecab node
        Returns:
            Boolean
        """
        # target_char = node.surface.decode("utf-8")
        target_char = node.surface
        if target_char[0].isdigit():
            return True
        else:
            return False

    @staticmethod
    def get_original(node, surface=False):
        """ get original
        Args:
            node: mecab node
            surface: flag to return surface only
        Returns:
            word of node
        Raises:
        """
        if surface is True:
            return node.surface.lower()
        else:
            if(len(node.feature.split(',')) >= INDEX_ORIGINAL_FORM and
               node.feature.split(',')[0] == "動詞"):
                return node.feature.split(',')[INDEX_ORIGINAL_FORM]
            elif(len(node.feature.split(',')) >= INDEX_ORIGINAL_FORM and
                 node.feature.split(',')[0] == "形容詞"):
                return node.feature.split(',')[INDEX_ORIGINAL_FORM]
            # elif(len(node.feature.split(',')) >= INDEX_ORIGINAL_FORM and
            #      node.feature.split(',')[0] == "名詞"):
            #     return node.feature.split(',')[INDEX_ORIGINAL_FORM]
            else:
                return node.surface.lower()
