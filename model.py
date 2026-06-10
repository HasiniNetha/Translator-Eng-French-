import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention
)

from tensorflow.keras import Model

from data import data

vocab_size = 1000
sequence_length = 10

english_sentences = [x[0] for x in data]
french_sentences = ["start " + x[1] + " end" for x in data]

source_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, sequence_length):
        super().__init__()

        self.token_emb = Embedding(
            vocab_size,
            embed_dim
        )

        self.pos_emb = Embedding(
            sequence_length,
            embed_dim
        )

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = self.token_emb(inputs)
        embedded_positions = self.pos_emb(positions)

        return embedded_tokens + embedded_positions


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dense_dim
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs):

        attention_output = self.attention(
            inputs,
            inputs
        )

        proj_input = self.layernorm1(
            inputs + attention_output
        )

        proj_output = self.dense_proj(
            proj_input
        )

        return self.layernorm2(
            proj_input + proj_output
        )


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()

    def call(
        self,
        inputs,
        encoder_outputs
    ):

        attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        attention_output2 = self.cross_attention(
            out1,
            encoder_outputs
        )

        out2 = self.layernorm2(
            out1 + attention_output2
        )

        ffn_output = self.ffn(out2)

        return self.layernorm3(
            out2 + ffn_output
        )


def train_model():

    source_vectorization.adapt(
        english_sentences
    )

    target_vectorization.adapt(
        french_sentences
    )

    encoder_input_data = source_vectorization(
        english_sentences
    )

    target_tokens = target_vectorization(
        french_sentences
    )

    decoder_input_data = target_tokens[:, :-1]
    decoder_targets = target_tokens[:, 1:]

    embed_dim = 128
    dense_dim = 256
    num_heads = 4

    encoder_input = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    x = PositionalEmbedding(
        vocab_size,
        embed_dim,
        sequence_length
    )(encoder_input)

    encoder_output = TransformerEncoder(
        embed_dim,
        num_heads,
        dense_dim
    )(x)

    decoder_inputs = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    x = PositionalEmbedding(
        vocab_size,
        embed_dim,
        sequence_length
    )(decoder_inputs)

    x = TransformerDecoder(
        embed_dim,
        dense_dim,
        num_heads
    )(x, encoder_output)

    decoder_output = Dense(
        vocab_size,
        activation="softmax"
    )(x)

    transformer = Model(
        [encoder_input, decoder_inputs],
        decoder_output
    )

    transformer.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    transformer.fit(
    [encoder_input_data, decoder_input_data],
    decoder_targets,
    epochs=300,
    batch_size=2,
    verbose=0
)

    return (
        transformer,
        source_vectorization,
        target_vectorization
    )


def translate(
    sentence,
    transformer,
    source_vectorization,
    target_vectorization
):

    encoder_input_test = source_vectorization(
        [sentence.lower()]
    )

    decoded_sentence = "start"

    for i in range(10):

        tokenized_target = target_vectorization(
            [decoded_sentence]
        )

        predictions = transformer.predict(
            [
                encoder_input_test,
                tokenized_target
            ],
            verbose=0
        )

        sampled_token_index = np.argmax(
            predictions[0, i, :]
        )

        vocab = target_vectorization.get_vocabulary()

        if sampled_token_index >= len(vocab):
            break

        sampled_token = vocab[
            sampled_token_index
        ]

        if sampled_token == "end":
            break

        decoded_sentence += " " + sampled_token

    return decoded_sentence.replace(
        "start",
        ""
    ).strip()