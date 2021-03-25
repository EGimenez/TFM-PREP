import numpy as np
#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import time
from tqdm import tqdm
from PIL import Image
from threading import Lock

lock = Lock()


def get(name):
    return tf.get_default_graph().get_tensor_by_name('import/' + name + ':0')


def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    return sess


optimized = False
if optimized:
    # Optimized model. Twice as fast as
    # 1. we freeze conditional network (label is always 0)
    # 2. we use fused kernels
    import blocksparse
    graph_path = 'graph_optimized.pb'
    inputs = {
        'dec_eps_0': 'dec_eps_0',
        'dec_eps_1': 'dec_eps_1',
        'dec_eps_2': 'dec_eps_2',
        'dec_eps_3': 'dec_eps_3',
        'dec_eps_4': 'dec_eps_4',
        'dec_eps_5': 'dec_eps_5',
        'enc_x': 'input/enc_x',
    }
    outputs = {
        'dec_x': 'model_3/Cast_1',
        'enc_eps_0': 'model_2/pool0/truediv_1',
        'enc_eps_1': 'model_2/pool1/truediv_1',
        'enc_eps_2': 'model_2/pool2/truediv_1',
        'enc_eps_3': 'model_2/pool3/truediv_1',
        'enc_eps_4': 'model_2/pool4/truediv_1',
        'enc_eps_5': 'model_2/truediv_4'
    }

    def update_feed(feed_dict, bs):
        return feed_dict
else:
    graph_path = 'graph_unoptimized.pb'
    inputs = {
        'dec_eps_0': 'Placeholder',
        'dec_eps_1': 'Placeholder_1',
        'dec_eps_2': 'Placeholder_2',
        'dec_eps_3': 'Placeholder_3',
        'dec_eps_4': 'Placeholder_4',
        'dec_eps_5': 'Placeholder_5',
        'enc_x': 'input/image',
        'enc_x_d': 'input/downsampled_image',
        'enc_y': 'input/label'
    }
    outputs = {
        'dec_x': 'model_1/Cast_1',
        'enc_eps_0': 'model/pool0/truediv_1',
        'enc_eps_1': 'model/pool1/truediv_1',
        'enc_eps_2': 'model/pool2/truediv_1',
        'enc_eps_3': 'model/pool3/truediv_1',
        'enc_eps_4': 'model/pool4/truediv_1',
        'enc_eps_5': 'model/truediv_4'
    }

    def update_feed(feed_dict, bs):
        x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
        y = np.zeros([bs], dtype=np.int32)
        feed_dict[enc_x_d] = x_d
        feed_dict[enc_y] = y
        return feed_dict

with tf.gfile.GFile(graph_path, 'rb') as f:
    graph_def_optimized = tf.GraphDef()
    graph_def_optimized.ParseFromString(f.read())

sess = tensorflow_session()
tf.import_graph_def(graph_def_optimized)

print("Loaded model")

n_eps = 6

# Encoder
enc_x = get(inputs['enc_x'])
enc_eps = [get(outputs['enc_eps_' + str(i)]) for i in range(n_eps)]
if not optimized:
    enc_x_d = get(inputs['enc_x_d'])
    enc_y = get(inputs['enc_y'])

# Decoder
dec_x = get(outputs['dec_x'])
dec_eps = [get(inputs['dec_eps_' + str(i)]) for i in range(n_eps)]

eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
              (16, 16, 48), (8, 8, 96), (4, 4, 384)]
eps_sizes = [np.prod(e) for e in eps_shapes]
eps_size = 256 * 256 * 3
z_manipulate = np.load('z_manipulate.npy')

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

flip_tags = ['No_Beard', 'Young']
for tag in flip_tags:
    i = _TAGS.index(tag)
    z_manipulate[i] = -z_manipulate[i]

scale_tags = ['Narrow_Eyes']
for tag in scale_tags:
    i = _TAGS.index(tag)
    z_manipulate[i] = 1.2*z_manipulate[i]

z_sq_norms = np.sum(z_manipulate**2, axis=-1, keepdims=True)
z_proj = (z_manipulate / z_sq_norms).T


def run(sess, fetches, feed_dict):
    with lock:
        # Locked tensorflow so average server response time to user is lower
        result = sess.run(fetches, feed_dict)
    return result


def flatten_eps(eps):
    # [BS, eps_size]
    return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)


def unflatten_eps(feps):
    index = 0
    eps = []
    bs = feps.shape[0]  # feps.size // eps_size
    for shape in eps_shapes:
        eps.append(np.reshape(
            feps[:, index: index+np.prod(shape)], (bs, *shape)))
        index += np.prod(shape)
    return eps


def encode(img):
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    bs = img.shape[0]
    assert img.shape[1:] == (256, 256, 3)
    feed_dict = {enc_x: img}

    update_feed(feed_dict, bs)  # For unoptimized model
    return flatten_eps(run(sess, enc_eps, feed_dict))


def decode(feps):
    if len(feps.shape) == 1:
        feps = np.expand_dims(feps, 0)
    bs = feps.shape[0]
    # assert len(eps) == n_eps
    # for i in range(n_eps):
    #     shape = (BATCH_SIZE, 128 // (2 ** i), 128 // (2 ** i), 6 * (2 ** i) * (2 ** (i == (n_eps - 1))))
    #     assert eps[i].shape == shape
    eps = unflatten_eps(feps)

    feed_dict = {}
    for i in range(n_eps):
        feed_dict[dec_eps[i]] = eps[i]

    update_feed(feed_dict, bs)  # For unoptimized model
    return run(sess, dec_x, feed_dict)


def project(z):
    return np.dot(z, z_proj)


def _manipulate(z, dz, alpha):
    z = z + alpha * dz
    return decode(z), z


def _manipulate_range(z, dz, points, scale):
    z_range = np.concatenate(
        [z + scale*(pt/(points - 1)) * dz for pt in range(0, points)], axis=0)
    return decode(z_range), z_range


# alpha from [0,1]
def mix(z1, z2, alpha):
    dz = (z2 - z1)
    return _manipulate(z1, dz, alpha)


def mix_range(z1, z2, points=5):
    dz = (z2 - z1)
    return _manipulate_range(z1, dz, points, 1.)


# alpha goes from [-1,1]
def manipulate(z, typ, alpha):
    dz = z_manipulate[typ]
    return _manipulate(z, dz, alpha)


def manipulate_all(z, typs, alphas):
    dz = 0.0
    for i in range(len(typs)):
        dz += alphas[i] * z_manipulate[typs[i]]
    return _manipulate(z, dz, 1.0)


def manipulate_range(z, typ, points=5, scale=1):
    dz = z_manipulate[typ]
    return _manipulate_range(z - dz, 2*dz, points, scale)


def random(bs=1, eps_std=0.7):
    feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
    return decode(feps), feps


def test():
    img = Image.open('test/img.png')
    img = np.reshape(np.array(img), [1, 256, 256, 3])

    # Encoding speed
    eps = encode(img)
    t = time.time()
    for _ in tqdm(range(10)):
        eps = encode(img)
    print("Encoding latency {} sec/img".format((time.time() - t) / (1 * 10)))

    # Decoding speed
    dec = decode(eps)
    t = time.time()
    for _ in tqdm(range(10)):
        dec = decode(eps)
    print("Decoding latency {} sec/img".format((time.time() - t) / (1 * 10)))
    img = Image.fromarray(dec[0])
    img.save('test/dec.png')

    # Manipulation
    dec, _ = manipulate(eps, _TAGS.index('Smiling'), 0.66)
    img = Image.fromarray(dec[0])
    img.save('test/smile.png')


def test2():
    from PIL import Image
    from numpy import linalg as LA

    img_eg = Image.open('test/032080.jpg')
    img_neg = Image.open('test/082099.jpg')

    old_size = img_eg.size
    new_size = (256, 256)

    new_img_eg = Image.new("RGB", new_size)
    new_img_neg = Image.new("RGB", new_size)

    # new_img_eg.paste(img_eg, ((new_size[0]-old_size[0])//2,
    #                           (new_size[1]-old_size[1])//2))
    #
    # new_img_neg.paste(img_neg, ((new_size[0]-old_size[0])//2,
    #                             (new_size[1]-old_size[1])//2))

    new_img_eg = img_eg.resize(new_size)
    new_img_neg = img_neg.resize(new_size)

    img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])
    img_neg = np.reshape(np.array(new_img_neg), [1, 256, 256, 3])

    eps_eg = encode(img_eg)
    eps_neg = encode(img_neg)

    for t in range(0,11):
        eps_t = eps_eg*(1-t/10) + eps_neg*t/10
        print(LA.norm(eps_t))
        dec = decode(eps_t)
        img = Image.fromarray(dec[0])
        img.save('test/resize_latent_lin'+str(t)+'.png')

    for t in range(0,11):
        eps_t = eps_eg*(1-t/10) + eps_neg*t/10
        eps_t = (LA.norm(eps_eg)*(1-t/10) + LA.norm(eps_neg)*t/10)*eps_t/LA.norm(eps_t)
        print(LA.norm(eps_t))
        dec = decode(eps_t)
        img = Image.fromarray(dec[0])
        img.save('test/resize_latent_norm'+str(t)+'.png')


    img_eg = np.array(new_img_eg)
    img_neg = np.array(new_img_neg)

    for t in range(0,11):
        eps_t = img_eg*(1-t/10) + img_neg*(t/10)
        img = Image.fromarray(np.uint8(eps_t))
        img.save('test/resize_original_lin_int'+str(t)+'.png')
        eps_t = np.reshape(eps_t, [1, 256, 256, 3])
        eps_t = encode(eps_t)
        print(LA.norm(eps_t))

    print('hola')

def test2_1():
    from PIL import Image
    from numpy import linalg as LA
    from PIL import ImageDraw as ImageDraw

    img_eg = Image.open('test/032080.jpg')
    img_neg = Image.open('test/032080.jpg')

    new_size = (256, 256)

    new_img_eg = img_eg.resize(new_size)
    new_img_neg = img_neg.resize(new_size)

    draw = ImageDraw.Draw(new_img_neg)
    x = 50
    y = 50
    r = 5
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))

    img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])
    img_neg = np.reshape(np.array(new_img_neg), [1, 256, 256, 3])

    eps_eg = encode(img_eg)
    eps_neg = encode(img_neg)

    for t in range(0,11):
        eps_t = eps_eg*(1-t/10) + eps_neg*t/10
        print(LA.norm(eps_t))
        dec = decode(eps_t)
        img = Image.fromarray(dec[0])
        img.save('test_2/resize_dot_latent_lin'+str(t)+'.png')

    print('hola')


def test3():
    import glob
    from PIL import Image
    from numpy import linalg as LA
    for my_img in glob.iglob(r'C:\Users\EGimenez\ME\projects\BGSE\TFM\img_align_celeba\img_align_celeba\*.jpg'):
        img_eg = Image.open(my_img)


    print('starting test 3')

    old_size = img_eg.size
    new_size = (256, 256)

    new_img_eg = Image.new("RGB", new_size)

    # new_img_eg.paste(img_eg, ((new_size[0]-old_size[0])//2,
    #                           (new_size[1]-old_size[1])//2))

    new_img_neg = img_eg.resize(new_size)
    img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])

    eps_eg = encode(img_eg)
    print(LA.norm(eps_eg))


def test_drop_components():
    from PIL import Image
    from numpy import linalg as LA

    img_eg = Image.open('test/032080.jpg')

    old_size = img_eg.size
    new_size = (256, 256)

    new_img_eg = Image.new("RGB", new_size)

    new_img_eg = img_eg.resize(new_size)

    img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])

    eps_eg = encode(img_eg)

    eps_eg_0 = eps_eg.copy()
    eps_eg_100 = eps_eg.copy()


    for t in range(0, 100):
        print(t)
        # Knokeamos 10 componentes
        for c in range(10):
            i = np.random.randint(eps_eg.shape[1])
            eps_eg_0[0, i] = 0
            eps_eg_100[0, i] = 100
        dec = decode(eps_eg_0)
        img = Image.fromarray(dec[0])
        img.save('test/drop_components_0_'+str(t)+'.png')
        dec = decode(eps_eg_100)
        img = Image.fromarray(dec[0])
        img.save('test/drop_components_100_'+str(t)+'.png')

    print('hola')

def test_drop_components_1_1():
    from PIL import Image
    from numpy import linalg as LA

    img_eg = Image.open('test/032080.jpg')

    old_size = img_eg.size
    new_size = (256, 256)

    new_img_eg = Image.new("RGB", new_size)

    new_img_eg = img_eg.resize(new_size)

    img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])

    eps_eg = encode(img_eg)


    for t in range(0, 1000):
        eps_eg_100 = eps_eg.copy()
        print(t)
        i = np.random.randint(eps_eg.shape[1])
        eps_eg_100[0, i] = 100
        dec = decode(eps_eg_100)
        img = Image.fromarray(dec[0])
        img.save('test/drop_components_100_1_1_i_100_'+str(i)+'.png')

    print('hola')


def test_drop_components_1_1_transition():
    from PIL import Image
    from numpy import linalg as LA

    # Modifications
    # 195309
    # 194582
    # 192426
    # 120327

    # People
    # 082099
    # 032080
    # img

    peoples = ['082099', '032080', 'img']
    components = [195309, 194582, 192426, 120327]

    for p in peoples:
        print(p)
        img_eg = Image.open('test/'+p+'.jpg')
        new_size = (256, 256)
        new_img_eg = img_eg.resize(new_size)
        img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])
        eps_eg = encode(img_eg)

        for c in components:
            print(c)
            eps_eg_100 = eps_eg.copy()

            for t in np.arange(-100, 101, 2):
                print(t)
                eps_eg_100[0, c] = t
                dec = decode(eps_eg_100)
                img = Image.fromarray(dec[0])
                img.save('test/transitions/components_p_'+str(p)+'_c_'+str(c)+'_t_'+str(t)+'.png')

    print('hola')



# warm start
_img, _z = random(1)
_z = encode(_img)
print("Warm started tf model")

if __name__ == '__main__':
    print('test_2')
    #test2()
    test_drop_components_1_1_transition()
    print('done')
    #test3()
