#import os
from re import X
import numpy as np
import tensorflow as tf
import pickle

# tf2
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, Input, Model, Sequential
from tensorflow.keras.layers import (Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dropout, Lambda, Input, Conv2D, 
    Concatenate, Add, Subtract, Embedding, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed, 
    Masking, Reshape, RepeatVector, Layer)
import os
import random


class MLP(Sequential):
    def __init__(self, input_shape, dims=(100, 50), activation='relu', last_activation=None):
        Sequential.__init__(self)
        if last_activation is None:
            last_activation = activation
        self.add(Dense(dims[0], activation=activation, input_shape=input_shape))
        for d in dims[1:-1]:
            self.add(Dense(d, activation=activation))
        self.add(Dense(dims[-1], activation=last_activation))
        

class MixtureOutput(Model):
    def __init__(self, input_shape, n, d=1, activation='relu', eps=1e-4, bias_mu=1.8, bias_sigma=0.2,
                 name=None):
        super().__init__()
        self.n = n
        self.d = d
        self.eps = eps
        self.alpha = Dense(n, activation='softmax')
        self.mu = Dense(n * d, activation=activation, bias_initializer=initializers.Constant(bias_mu))

        self.sigma = Dense(n * d, activation='relu', bias_initializer=initializers.Constant(bias_sigma))

    def call(self, inp_masked):
        inp = StripMask()(inp_masked)
        alpha = self.alpha(inp)
        alpha = Reshape((self.n, 1))(alpha)
        mu = self.mu(inp)
        mu = Reshape((self.n, self.d))(mu)
        sigma = self.sigma(inp)
        sigma = Lambda(lambda x: x + self.eps)(sigma)  # Add epsilon to avoid division by 0
        sigma = Reshape((self.n, self.d))(sigma)

        out = Concatenate(axis=2)([alpha, mu, sigma])
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.n,3)
        


class NormalizedScaleEmbedding(Layer):
    def __init__(self, input_shape, activation='relu', downsample=1, mlp_dims=(500, 300, 200, 150), eps=1e-8):
        super().__init__()
        self.supports_masking = True
        self.activation = activation
        self.inp_shape = input_shape
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps
        self.cnn = Sequential()
        self.cnn.add(Conv2D(8, (downsample, 1), strides=(downsample, 1), activation=activation))
        self.cnn.add(Conv2D(32, (16, 3), strides=(1, 3), activation=activation))
        self.cnn.add(Reshape((-1, 32 * self.inp_shape[-1] // 3)))
        self.cnn.add(Conv1D(64, 16, activation=activation))
        self.cnn.add(MaxPooling1D(2))
        self.cnn.add(Conv1D(128, 16, activation=activation))
        self.cnn.add(MaxPooling1D(2))
        self.cnn.add(Conv1D(32, 8, activation=activation))
        self.cnn.add(MaxPooling1D(2))
        self.cnn.add(Conv1D(32, 8, activation=activation))
        self.cnn.add(Conv1D(16, 4, activation=activation))
        self.cnn.add(Flatten())
        self.mlp = MLP(input_shape=(865,), dims=self.mlp_dims, activation=activation)

    def call(self, inputs):
        activation = self.activation
        downsample = self.downsample
        x = Lambda(lambda t: t / (K.max(K.abs(t), axis=(1, 2), keepdims=True) + self.eps))(inputs)
        x = Lambda(lambda t: K.expand_dims(t))(x)
        scale = Lambda(lambda t: K.log(K.max(K.abs(t), axis=(1, 2)) + self.eps) / 100)(inputs)
        scale = Lambda(lambda t: K.expand_dims(t))(scale)
        x = self.cnn(x)
        x = Concatenate()([x, scale])
        x = self.mlp(x)
        return x

    def compute_output_shape(self, input_shape):
        return (None, 500)

  
class Transformer(Model):
    def __init__(self, max_stations=32, emb_dim=500, layers=6, att_masking=False, hidden_dropout=0.0,
                 mad_params={}, ffn_params={}, norm_params={}):
        super().__init__()  
        self.blocks = [(MultiHeadSelfAttention(**mad_params),
                        PointwiseFeedForward(**ffn_params),
                        LayerNormalization(**norm_params),
                        LayerNormalization(**norm_params))
                       for _ in range(layers)]
        self.att_masking = att_masking
        self.hidden_dropout = hidden_dropout
    def call(self,inp,att_mask=None):

        x = inp
        for attention_layer, ffn_layer, norm1_layer, norm2_layer in self.blocks:
            if att_mask is not None:
                modified_x = attention_layer([x, att_mask])
            else:
                modified_x = attention_layer(x)

            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)


            x = norm1_layer(Add()([x, modified_x]))
            modified_x = ffn_layer(x)
            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)

            x = norm2_layer(Add()([x, modified_x]))
        
        return x

class Conformer(Model):
    def __init__(self, max_stations=32, emb_dim=500, layers=6, att_masking=False, hidden_dropout=0.0,
                 mad_params={}, ffn_params={}, norm_params={}):
        super().__init__()  
        
        self.blocks = [(PointwiseFeedForward(**ffn_params),
                        MultiHeadSelfAttention(**mad_params),
                        ConvModule(input_dim=emb_dim),
                        PointwiseFeedForward(**ffn_params),
                        LayerNormalization(**norm_params),
                        )
                       for _ in range(layers)]
        self.att_masking = att_masking
        self.hidden_dropout = hidden_dropout
    def call(self,inp,att_mask=None):

        x = inp
        
        for ffn_layer_1 ,attention_layer,convm, ffn_layer_2, norm1_layer in self.blocks:
            
            x = x + ffn_layer_1(x)/2
            
            if att_mask is not None:
                modified_x = attention_layer([x, att_mask])
            else:
                modified_x = attention_layer(x)

            modified_x = convm(modified_x + x)
            
            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)

            modified_x = modified_x + ffn_layer_2(x)/2
            
            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)
                
            x = norm1_layer(modified_x)
        return x
        
L2 = tf.keras.regularizers.l2(1e-6)
def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout=0.0,
        depth_multiplier=1,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=2 * input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=1,
            padding="same",
            name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish,
            name=f"{name}_swish_activation",
        )
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.ln(inputs, training=training)
        B, T, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class GLU(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=-1,
        name="glu_activation",
        **kwargs,
    ):
        super(GLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(
        self,
        inputs,
        **kwargs,
    ):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf
# Calculates and concatenates sinusoidal embeddings for lat, lon and depth
# Note: Permutation is completely unnecessary, but kept for compatibility reasons
# WARNING: Does not take into account curvature of the earth!


class PositionEmbedding(Layer):
    def __init__(self, wavelengths, emb_dim, borehole=False, rotation=None, rotation_anchor=None, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.wavelengths = wavelengths  # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            # print(f'Rotating by {np.rad2deg(rotation)} degrees')
            c, s = np.cos(rotation), np.sin(rotation)
            self.rotation_matrix = K.variable(np.array(((c, -s), (s, c))), dtype=K.floatx())
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        assert emb_dim % 10 == 0
        if borehole:
            assert emb_dim % 20 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        if borehole:
            depth_dim = emb_dim // 20
        self.lat_coeff = 2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        self.lon_coeff = 2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        self.depth_coeff = 2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.mask = self.mask.astype('int32')
        self.fake_borehole = False

    def build(self, input_shape):
        if input_shape[-1] == 3:
            self.fake_borehole = True
        super(PositionEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1] 
            lon_base *= K.cos(lat_base * np.pi / 180)

            lat_base -= self.rotation_anchor[0]
            lon_base -= self.rotation_anchor[1] * K.cos(self.rotation_anchor[0] * np.pi / 180)

            latlon = K.stack([lat_base, lon_base], axis=-1)
            rotated = latlon @ self.rotation_matrix

            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff
            lon_base = x[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        if self.borehole:
            if self.fake_borehole:
                # Use third value for the depth of the top station and 0 for the borehole depth
                depth_base = x[:, :, 2:3] * self.depth_coeff * 0
                depth2_base = x[:, :, 2:3] * self.depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * self.depth_coeff
            output = tf.concat([K.sin(lat_base), K.cos(lat_base),
                                K.sin(lon_base), K.cos(lon_base),
                                K.sin(depth_base), K.cos(depth_base),
                                K.sin(depth2_base), K.cos(depth2_base)], axis=-1)
        else:
            output = tf.concat([K.sin(lat_base), K.cos(lat_base),
                                K.sin(lon_base), K.cos(lon_base),
                                K.sin(depth_base), K.cos(depth_base)], axis=-1)
        output = tf.gather(output, self.mask, axis=-1)
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            output *= mask  # Zero out all masked elements
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.emb_dim,)

    def compute_mask(self, inputs, mask=None):
        return mask


class MultiHeadSelfAttention(Layer):
    def __init__(self, n_heads, infinity=1e6,
                 att_masking=False,
                 kernel_initializer=keras.initializers.RandomUniform(minval=-1.2, maxval=1.2),
                 att_dropout=0.0,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.infinity = infinity
        # Attention masking: Model may only attend to stations where attention mask is true
        # Different from regular masking, as masked (i.e. att_mask = False) stations still collect information
        self.att_masking = att_masking
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.att_dropout = att_dropout

    def build(self, input_shape,att_mask=None):
        
        if self.att_masking:
            input_shape = input_shape[0]
        n_heads = self.n_heads
        d_model = input_shape[-1]  # Embedding dim
        self.stations = input_shape[1]
        assert d_model % n_heads == 0
        d_key = d_model // n_heads  # = d_query = d_val
        self.d_key = d_key
        self.WQ = self.add_weight('WQ', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WK = self.add_weight('WK', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WV = self.add_weight('WV', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WO = self.add_weight('WO', (d_key * n_heads, d_model), initializer=self.kernel_initializer)
        super(MultiHeadSelfAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        d_key = self.d_key
        n_heads = self.n_heads
        if self.att_masking:
            att_mask = x[1]
            x = x[0]
            if mask is not None:
                mask = mask[0]
        else:
            att_mask = None
            
        q = K.dot(x, self.WQ)  # (batch, stations, key*n_heads)
        q = K.reshape(q, (-1, self.stations, d_key, n_heads))
        q = K.permute_dimensions(q, [0, 3, 1, 2])  # (batch, n_heads, stations, key)
        k = K.dot(x, self.WK)  # (batch, stations, key*n_heads)
        k = K.reshape(k, (-1, self.stations, d_key, n_heads))
        k = K.permute_dimensions(k, [0, 3, 2, 1])  # (batch, n_heads, key, stations)
        score = tf.matmul(q, k) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)
        if mask is not None:
            inv_mask = K.expand_dims(K.expand_dims(K.cast(~mask, K.floatx()), axis=-1), axis=-1)  # (batch, stations, 1, 1)
            mask_B = K.permute_dimensions(inv_mask, [0, 2, 3, 1])  # (batch, 1, 1, stations)
            score = score - mask_B * self.infinity
        if att_mask is not None:
            inv_mask = K.expand_dims(K.expand_dims(K.cast(~att_mask, K.floatx()), axis=-1),
                                     axis=-1)  # (batch, stations, 1, 1)
            mask_B = K.permute_dimensions(inv_mask, [0, 2, 3, 1])  # (batch, 1, 1, stations)
            score = score - mask_B * self.infinity
        score = K.softmax(score)
        if self.att_dropout > 0:
            score = K.dropout(score, self.att_dropout)
        v = K.dot(x, self.WV)  # (batch, stations, key*n_heads)
        v = K.reshape(v, (-1, self.stations, d_key, n_heads))
        v = K.permute_dimensions(v, [0, 3, 1, 2])  # (batch, n_heads, stations, key)
        o = tf.matmul(score, v)  # (batch, n_heads, stations, key)
        o = K.permute_dimensions(o, [0, 2, 1, 3])  # (batch, stations, n_heads, key)
        o = K.reshape(o, (-1, self.stations, n_heads * d_key))
        o = K.dot(o, self.WO)
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            o = K.abs(o * mask)
        return o

    def compute_output_shape(self, input_shape):
        if self.att_masking:
            return input_shape[0]
        else:
            return input_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        if self.att_masking:
            return mask[0]
        else:
            return mask


class PointwiseFeedForward(Layer):
    def __init__(self, hidden_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(PointwiseFeedForward, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel1 = self.add_weight('kernel1', (input_shape[-1], self.hidden_dim), initializer=self.kernel_initializer)
        self.bias1 = self.add_weight('bias1', (self.hidden_dim,), initializer=self.bias_initializer)
        self.kernel2 = self.add_weight('kernel2', (self.hidden_dim, input_shape[-1]), initializer=self.kernel_initializer)
        self.bias2 = self.add_weight('bias2', (input_shape[-1],), initializer=self.bias_initializer)
        super(PointwiseFeedForward, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        x = gelu(K.dot(x, self.kernel1) + self.bias1)
        x = K.dot(x, self.kernel2) + self.bias2
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            x *= mask  # Zero out all masked elements
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerNormalization(Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.beta = self.add_weight('beta', input_shape[-1:], initializer=keras.initializers.Zeros())
        self.gamma = self.add_weight('gamma', input_shape[-1:], initializer=keras.initializers.Ones())
        super(LayerNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        # Axis according to https://github.com/tensorflow/tensor2tensor/blob/05f222d27a4885550450d9ba26987f78af5f9ecd/tensor2tensor/layers/common_layers.py#L705
        m = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - m), axis=-1, keepdims=True)
        z = (x - m) / K.sqrt(s + self.eps)
        output = self.gamma * z + self.beta
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            output *= mask  # Zero out all masked elements
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class AddEventToken(Layer):
    def __init__(self, fixed=True, init_range=None, **kwargs):
        # If fixed: Use 1 as constant to ensure that the attention in the first layer works properly
        # Else: Use learnable event token initialized to ones
        super(AddEventToken, self).__init__(**kwargs)
        self.fixed = fixed
        self.emb = None
        self.init_range = init_range

    def build(self, input_shape):
        if not self.fixed:
            if self.init_range is None:
                initializer = keras.initializers.Ones()
            else:
                initializer = keras.initializers.RandomUniform(minval=-self.init_range, maxval=self.init_range)
            self.emb = self.add_weight('emb', (input_shape[2],), initializer=initializer)
        super(AddEventToken, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        pad = K.ones_like(x[:, :1, :])
        if self.emb is not None:
            pad *= self.emb
        x = K.concatenate([pad, x], axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + 1, input_shape[2]

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return tf.pad(tensor=mask, paddings=[[0, 0], [1, 0]], mode='CONSTANT', constant_values=True)


class AddConstantToMixture(Layer):
    def __init__(self, **kwargs):
        super(AddConstantToMixture, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddConstantToMixture, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        mix, const = x
        const = K.expand_dims(const, axis=-1)
        alpha = tf.gather(mix, 0, axis=-1)
        mu = tf.gather(mix, 1, axis=-1) + const
        sigma = tf.gather(mix, 2, axis=-1)
        output = K.stack([alpha, mu, sigma], axis=-1)
        mask = self.compute_mask(x, mask)
        if mask is not None:
            mask = K.cast(mask, dtype=K.floatx())
            while mask.ndim < output.ndim:
                mask = K.expand_dims(mask, -1)
            output *= mask
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        else:
            mask1 = mask[0]
            mask2 = mask[1]
            if mask1 is None:
                return mask2
            elif mask2 is None:
                return mask1
            else:
                return tf.logical_and(mask1, mask2)


class Masking_nd(Layer):
    def __init__(self, mask_value=0., axis=-1, nodim=False, **kwargs):
        super(Masking_nd, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def compute_mask(self, inputs, mask=None):
        if self.nodim:
            output_mask = K.not_equal(inputs, self.mask_value)
        else:
            output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=self.axis)
        return output_mask

    def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs, self.mask_value),
                             axis=self.axis, keepdims=True)
        return inputs * K.cast(boolean_mask, K.dtype(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape


class GetMask(Layer):
    def __init__(self, **kwargs):
        super(GetMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GetMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def compute_mask(self, inputs, mask=None):
        return mask


class StripMask(Layer):
    def __init__(self, **kwargs):
        super(StripMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StripMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None


# From: https://github.com/openai/gpt-2/blob/ac5d52295f8a1c3856ea24fb239087cc1a3d1131/src/model.py#L25
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def mixture_density_loss(y_true, y_pred, eps=1e-6, d=1, mean=True, print_shapes=True):
    # if print_shapes:
    #     print(f'True: {y_true.shape}')
    #     print(f'Pred: {y_pred.shape}')
    alpha = y_pred[:, :, 0]
    density = K.ones_like(y_pred[:, :, 0])  # Create an array of ones of correct size
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = K.maximum(sigma, eps)
        density *= 1 / (np.sqrt(2 * np.pi) * sigma) * K.exp(-(y_true[:, j] - mu) ** 2 / (2 * sigma ** 2))
    density *= alpha
    density = K.sum(density, axis=1)
    density += eps
    loss = - K.log(density)
    if mean:
        return tf.reduce_mean(loss)
        return K.mean(loss)
    else:
        return loss


def time_distributed_loss(y_true, y_pred, loss_func, norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = K.reshape(y_true, (-1, (y_pred.shape[-1] - 1) // 2, 1))
    y_pred = K.reshape(y_pred, (-1, y_pred.shape[-2], y_pred.shape[-1]))
    loss = loss_func(y_true, y_pred, **kwloss)
    loss = K.reshape(loss, (-1, seq_length))

    if mean:
        return K.mean(loss)

    loss /= norm
    if summation:
        loss = K.sum(loss)

    return loss


class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def call(self, x, mask=None):
        pseudo_infty = 1000
        if mask is None:
            # Ensure that the mask is not the maximum value any more
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            x = x - mask * pseudo_infty
            return K.max(x, axis=1)
        else:
            return super().call(x)

    def compute_mask(self, inputs, mask=None):
        return None

class myModel(Model):
    def __init__(self, 
                max_stations,
                model="transformer",
                waveform_model_dims=(500, 500, 500),
                output_mlp_dims=(150, 100, 50, 30, 10),
                output_location_dims=(150, 100, 50, 50, 50),
                wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                mad_params={"n_heads": 10,
                            "att_dropout": 0.0,
                            "initializer_range": 0.02},
                ffn_params={'hidden_dim': 1000},
                transformer_layers=6,
                hidden_dropout=0.0,
                activation='relu',
                n_pga_targets=0,
                location_mixture=5,
                pga_mixture=5,
                magnitude_mixture=5,
                borehole=False,
                bias_mag_mu=1.8,
                bias_mag_sigma=0.2,
                bias_loc_mu=0,
                bias_loc_sigma=1,
                event_token_init_range=None,
                dataset_bias=False,
                n_datasets=None,
                no_event_token=False,
                trace_length=3000,
                downsample=5,
                rotation=None,
                rotation_anchor=None,
                skip_transformer=False,
                alternative_coords_embedding=False,
                single_model=None,
                **kwargs):
        super().__init__()
        if kwargs:
            print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')

        self.emb_dim = waveform_model_dims[-1]
        mad_params = mad_params.copy()  # Avoid modifying the input dicts
        ffn_params = ffn_params.copy()


        self.alternative_coords_embedding = alternative_coords_embedding
        self.skip_transformer = skip_transformer
        self.no_event_token = no_event_token
        self.event_token_init_range = event_token_init_range
        self.n_pga_targets = n_pga_targets
        self.dataset_bias = dataset_bias
        self.max_stations = max_stations
        self.borehole = borehole
        self.trace_length = trace_length
        
        if 'initializer_range' in mad_params:
            r = mad_params['initializer_range']
            mad_params['kernel_initializer'] = keras.initializers.RandomUniform(minval=-r, maxval=r)
            del mad_params['initializer_range']

        #   Single station model
        if borehole:
            input_shape = (trace_length, 6)
            self.metadata_shape = (4,)
        else:
            input_shape = (trace_length, 3)
            self.metadata_shape = (3,)

        
        self.waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
                                                    mlp_dims=waveform_model_dims)
        
        
        # if (single_model == None):
        #     self.waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
        #                                             mlp_dims=waveform_model_dims)
        # else:
        #     print("load pre train single model")
        #     self.waveform_model = single_model.layers[1]
        
        self.layer_norm = LayerNormalization()    
        
        if n_pga_targets:
            att_masking = True
            mad_params['att_masking'] = True
        else:
            att_masking = False
            mad_params['att_masking'] = False

        if not no_event_token:
            transformer_max_stations = max_stations + 1 + n_pga_targets
        else:
            transformer_max_stations = max_stations + n_pga_targets


        if not alternative_coords_embedding:
            self.coords_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=self.emb_dim , borehole=borehole,
                                        rotation=rotation, rotation_anchor=rotation_anchor)
            
        if self.n_pga_targets:
            self.pga_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=self.emb_dim , borehole=borehole,
                                        rotation=rotation, rotation_anchor=rotation_anchor)
        else:
            if self.skip_transformer:
                mlp_input_length = self.emb_dim
                if self.alternative_coords_embedding:
                    mlp_input_length += self.metadata_shape[0]
                self.mlp = MLP((mlp_input_length,), [self.emb_dim, self.emb_dim], activation=activation)
                
                
        if not skip_transformer:
            print(f"model:{model}")
            if(model=="transformer"):
                self.transformer = Transformer(max_stations=transformer_max_stations, emb_dim=self.emb_dim, att_masking=att_masking,
                                        layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                        ffn_params=ffn_params)
            if(model=="conformer"):
                self.transformer = Conformer(max_stations=transformer_max_stations, emb_dim=self.emb_dim, att_masking=att_masking,
                                    layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                    ffn_params=ffn_params)

        self.w = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(max_stations, 2)),dtype=tf.float32)
        self.w_layer = tf.keras.layers.Dense(2,use_bias=False,activation="softmax")
        self.mlp_pga = MLP((self.emb_dim ,), output_mlp_dims, activation=activation)

        self.output_model_pga = MixtureOutput((output_mlp_dims[-1],), pga_mixture, activation='linear', bias_mu=-5, \
                                                bias_sigma=1,name="pga")
    def build(self, input_shape):
        
        super(myModel, self).build(input_shape)  # Be sure to call this at the end          
    def call(self, inputs):
        
        waveform_inp,metadata_inp,pga_targets_inp = inputs

        waveforms_masked = Masking_nd(0, (2, 3))(waveform_inp)
        coords_masked = Masking(0)(metadata_inp)
        
        waveforms_emb = TimeDistributed(self.waveform_model)(waveforms_masked)
        waveforms_emb = self.layer_norm(waveforms_emb)

        
        if not self.alternative_coords_embedding:
            coords_emb = self.coords_emb(coords_masked)
            
            

            w = self.w_layer(self.w)
            
            w = tf.reshape(tf.tile(tf.expand_dims(w, axis=2),[1,1,500]),(2,25,500))

            a = tf.multiply(w[0],waveforms_emb)
            b = tf.multiply(w[1],coords_emb)

            # print(w[0,:,0])
            emb = Add()([a,b])
            
            
            # emb = Add()([waveforms_emb, coords_emb])
        else:
            emb = Concatenate(axis=-1)([waveforms_emb, coords_masked])

        
        if not (self.skip_transformer or self.no_event_token):
            emb = AddEventToken(fixed=False, init_range=self.event_token_init_range)(emb)

        

        if self.n_pga_targets:
            pga_targets_masked = Masking(0)(pga_targets_inp)
            
            pga_emb = self.pga_emb(pga_targets_masked)
            att_mask = K.concatenate([K.ones_like(emb[:, :, 0], dtype=bool),
                                    K.zeros_like(pga_emb[:, :, 0], dtype=bool)], axis=1)
            
            emb = tf.concat([emb, pga_emb],axis=1)
            # print(f"emb.shape:{emb.shape}")
            # emb = Concatenate(axis=1)([pga_emb, pga_emb])
            
            emb = self.transformer(emb, att_mask)
        
        
        else:
            if self.skip_transformer:
                mlp_input_length = self.emb_dim
                if self.alternative_coords_embedding:
                    mlp_input_length += self.metadata_shape[0]

                emb = TimeDistributed(self.mlp)(emb)
                emb = GlobalMaxPooling1DMasked()(emb)
            else:
                emb = self.transformer(emb)
        
        
        if not self.no_event_token:
            if self.skip_transformer:
                event_emb = emb
            else:
                event_emb = Lambda(lambda x: x[:, 0, :])(emb)  # Select event embedding

            mag_embedding = self.mlp_mag(event_emb)
            out = self.output_model(mag_embedding)

            loc_embedding = self.mlp_loc(event_emb)
            out_loc = self.output_model_loc(loc_embedding)

        if self.n_pga_targets:
            pga_emb = Lambda(lambda x: x[:, -self.n_pga_targets:, :])(emb)  # Select embeddings for pga
            pga_emb = TimeDistributed(self.mlp_pga,input_shape=(64,self.emb_dim))(pga_emb)
            
            output_pga = TimeDistributed(self.output_model_pga)(pga_emb)
        
        return {"pga": output_pga}
    
        
        
        
        
    
def build_transformer_model(max_stations,
                            waveform_model_dims=(500, 500, 500),
                            output_mlp_dims=(150, 100, 50, 30, 10),
                            output_location_dims=(150, 100, 50, 50, 50),
                            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                            mad_params={"n_heads": 10,
                                        "att_dropout": 0.0,
                                        "initializer_range": 0.02
                                        },
                            ffn_params={'hidden_dim': 1000},
                            transformer_layers=6,
                            hidden_dropout=0.0,
                            activation='relu',
                            n_pga_targets=0,
                            location_mixture=5,
                            pga_mixture=5,
                            magnitude_mixture=5,
                            borehole=False,
                            bias_mag_mu=1.8,
                            bias_mag_sigma=0.2,
                            bias_loc_mu=0,
                            bias_loc_sigma=1,
                            event_token_init_range=None,
                            dataset_bias=False,
                            n_datasets=None,
                            no_event_token=False,
                            trace_length=3000,
                            downsample=5,
                            rotation=None,
                            rotation_anchor=None,
                            skip_transformer=False,
                            alternative_coords_embedding=False,
                            config = None,
                            **kwargs):


    mad_params = mad_params.copy()  # Avoid modifying the input dicts
    ffn_params = ffn_params.copy()

    if 'initializer_range' in mad_params:
        r = mad_params['initializer_range']
        mad_params['kernel_initializer'] = keras.initializers.RandomUniform(minval=-r, maxval=r)
        del mad_params['initializer_range']

    #   Single station model
    if borehole:
        input_shape = (trace_length, 6)
        metadata_shape = (4,)
    else:
        input_shape = (trace_length, 3)
        metadata_shape = (3,)
        
    waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
                                              mlp_dims=waveform_model_dims)
    
    mlp_mag_single_station = MLP((500,), output_mlp_dims, activation=activation)
    output_model_single_station = MixtureOutput((output_mlp_dims[-1],), 5, name='magnitude',
                                                bias_mu=bias_mag_mu, bias_sigma=bias_mag_sigma)

    waveform_inp_single_station = Input(shape=input_shape)
    emb = waveform_model(waveform_inp_single_station)
    emb = mlp_mag_single_station(emb)
    out = output_model_single_station(emb)

    single_station_model = Model(waveform_inp_single_station, out)
    full_model = myModel(**config['model_params'],trace_length=trace_length,single_model=single_station_model)

    return single_station_model, full_model


class EnsembleEvaluateModel:
    def __init__(self, config, max_ensemble_size=None, loss_limit=None,generators=None):
        self.config = config
        self.ensemble = config.get('ensemble', 1)
        true_ensemble_size = self.ensemble

        # self.ensemble = 1
        if max_ensemble_size is not None:
            self.ensemble = min(self.ensemble, max_ensemble_size)
        self.models = []
        for ens_id in range(self.ensemble):
            # ens_id = 4
            model_params = config['model_params'].copy()
            if config['training_params'].get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                model_params['rotation'] = np.pi / 4 * ens_id / (true_ensemble_size - 1)
            
            _, model = build_transformer_model(**config['model_params'],
                                                                        trace_length=3000,config=config)
            # model.compile(run_eagerly = True)
            model.build([(64,25,3000,3),generators[0][0][1].shape,generators[0][0][2].shape])
            self.models += [model]

        self.loss_limit = loss_limit

    def predict_generator(self, generator, **kwargs):
        preds = [model.predict_generator(generator, **kwargs) for model in self.models]
        return self.merge_preds(preds)

    def predict(self, inputs):
        preds = [model.predict(inputs) for model in self.models]
        return self.merge_preds(preds)

    @staticmethod
    def merge_preds(preds):
        merged_preds = []

        if isinstance(preds, list):
            iter = range(len(preds[0]))
        else:
            iter = [-1]

        for i in iter:  # Iterate over mag, loc, pga, ...
            if i != -1:
                pred_item = np.concatenate([x['pga'] for x in preds], axis=-2)
            else:
                pred_item = np.concatenate(preds['pga'], axis=-2)
                
            if len(pred_item.shape) == 3:
                pred_item[:, :, 0] /= np.sum(pred_item[:, :, 0], axis=-1, keepdims=True)
            elif len(pred_item.shape) == 4:
                pred_item[:, :, :, 0] /= np.sum(pred_item[:, :, :, 0], axis=-1, keepdims=True)
            else:
                raise ValueError("Encountered prediction of unexpected shape")
            merged_preds += [pred_item]

        if len(merged_preds) == 1:
            return merged_preds[0]
        else:
            return merged_preds

    def load_weights(self, weights_path):
        tmp_models = self.models
        self.models = []
        removed_models = 0
        for ens_id, model in enumerate(tmp_models):
            # ens_id = 4
            if self.loss_limit is not None:
                hist_path = os.path.join(weights_path, f'{ens_id}', 'hist.pkl')
                with open(hist_path, 'rb') as f:
                    hist = pickle.load(f)
                if np.min(hist['val_loss']) > self.loss_limit:
                    removed_models += 1
                    continue

            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')
            weight_file = sorted([x for x in os.listdir(tmp_weights_path) if x[:5] == 'event'])[-1]
            weight_file = os.path.join(tmp_weights_path, weight_file)
            print(weight_file)
            model.load_weights(weight_file)
            self.models += [model]

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')