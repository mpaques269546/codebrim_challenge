import os
import torch
import torch.nn as nn


from encoder.vision_transformer import VisionTransformer
from decoder.classifier import LinearClassifier




def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name):
    
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        #print(state_dict.keys())
        if checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        #print('pretrained=', state_dict.keys())
        #print('model sate dict=', model.state_dict().keys())

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `model.` prefix induced by saving models
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
        # in case different inp size: change pos_embed 
        if "pos_embed" in state_dict.keys():
            pretrained_shape = state_dict['pos_embed'].size()[1]
            model_shape = model.state_dict()['pos_embed'].size()[1]
            if pretrained_shape != model_shape:
                pos_embed = state_dict['pos_embed']
                pos_embed = pos_embed.permute(0, 2, 1)
                pos_embed = F.interpolate(pos_embed, size=model_shape)
                pos_embed = pos_embed.permute(0, 2, 1)
                state_dict['pos_embed'] = pos_embed
        #ignore linear layer
        #if "linear.bias" in state_dict.keys():
        #del state_dict['linear.weight'], state_dict['linear.bias']

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    else:
        print('ERROR: file {0} does not exist'.format(pretrained_weights))
        print('We use random weights.')


def build_encoder( pretrained_weights='', key='',  arch=None, patch_size=8 , avgpool=False, image_size=224, drop_rate=0, trainable=False,):
    
    arch_dic = {'vit_tiny':{ 'd_model':384, 'n_heads':3, 'n_layers':12},
      'vit_small':{ 'd_model':384, 'n_heads':6, 'n_layers':12},
      'vit_base':{'d_model':384, 'n_heads':12, 'n_layers':12},
      'vit_large':{ 'd_model':384, 'n_heads':24, 'n_layers':12},}

    if arch in arch_dic.keys():
        d_model = arch_dic[arch]['d_model']
        n_heads = arch_dic[arch]['n_heads']
        n_layers = arch_dic[arch]['n_layers']
        n_cls = 1 #don't care only usefull for classification head
        # image_size=
        model = VisionTransformer(img_size=[image_size], patch_size=patch_size, in_chans=3, num_classes=n_cls, embed_dim=d_model, depth=n_layers,
                 num_heads=n_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm)
       
       

    else:
        print(f"Unknow architecture: {arch}")
        sys.exit(1)

    # load weights to evaluate
    if len(pretrained_weights)>0:
        load_pretrained_weights(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')

    # params
    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable encoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen encoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{arch} adapter built. {ct} trainable params, {cf} frozen params.")
   
    return model


def build_decoder(pretrained_weights, key,   num_cls=2, embed_dim=384*4, image_size=224, activation=None, trainable=False):
    
    model = LinearClassifier( embed_dim=embed_dim, num_cls=num_cls,  activation=activation )
    
    # load weights to evaluate
    if len(key)>0 and model!=None:
        arch=""
        load_pretrained_weights(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')

    # params
    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable decoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen decoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{key} adapter built. {ct} trainable params, {cf} frozen params.")
    return model


       

class Ensemble(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        trainable_encoder=False,
        trainable_decoder=False,
        n_last_blocks=4,
        avgpool_patchtokens=False
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trainable_encoder = trainable_encoder
        self.trainable_decoder = trainable_decoder
        self.n_last_blocks= n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens


    def forward(self, inp, return_features=False):
        ## encoder
        # final_features = [B, D]
        if self.trainable_encoder:
            features = self.encoder.forward_classification(inp, n=self.n_last_blocks, avgpool=self.avgpool_patchtokens)
        else:
            with torch.no_grad():
                features = self.encoder.forward_classification(inp, n=self.n_last_blocks, avgpool=self.avgpool_patchtokens)
                features = features.detach()
        
        ## decoder
        # output = [B, num_cls]     
        if self.trainable_decoder:
            output = self.decoder.forward(features )
        else:
            with torch.no_grad():
                output = self.decoder.forward(features)
        if return_features:
            return [features, output]
       
        return  output



def build_model(pretrained_weights, img_size=224, num_cls=6):
    encoder = build_encoder(pretrained_weights, arch='vit_small',  key='encoder', image_size=img_size)
    n_last_blocks = 4
    avgpool_patchtokens = False
    embed_dim = encoder.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

    decoder = build_decoder(pretrained_weights, key='decoder', num_cls=num_cls, 
                            embed_dim=embed_dim, image_size=img_size, activation=nn.Sigmoid())

    model = Ensemble(encoder, decoder,n_last_blocks= n_last_blocks, avgpool_patchtokens=avgpool_patchtokens)
    return model