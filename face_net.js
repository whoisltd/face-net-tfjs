const tf = require('@tensorflow/tfjs-node')
const mtcnn = require('@whoisltd/mtcnn-tfjs')
const sharp = require('sharp')

class Lambda extends tf.layers.Layer {
    static className = 'Lambda'
    constructor(config) {
        super(config)
        this.config = config
    }
    build(inputShape) {
        this.built = true
    }
    call(inputs, kwargs) {
        const {scale} = this.config.arguments
        const [x] = inputs
        const y = tf.mul(x, scale)
        return y
    }
    
}

tf.serialization.registerClass(Lambda)

async function load_model(url){
    const model = await tf.loadLayersModel(url)
    return model
}

async function img_to_endcoding(image){
    
    const model = await load_model('file://facenet512_model/model.json')
    factor_0 = tf.div(160, image.shape[0])
    factor_1 = tf.div(160, image.shape[1])

    factor = Math.min(factor_0.dataSync()[0], factor_1.dataSync()[0])
    
    dsize0 = tf.cast(tf.round(tf.mul(image.shape[1], factor)), 'int32')
    dsize1 =  tf.cast(tf.round(tf.mul(image.shape[0], factor)), 'int32')

    resized = tf.image.resizeBilinear(image, [dsize0.dataSync()[0], dsize1.dataSync()[0]])

    diff_0 = tf.sub(160, resized.shape[0])
    diff_1 = tf.sub(160, resized.shape[1])

    img = tf.pad(resized, [[tf.floorDiv(diff_0,2), tf.sub(diff_0, tf.floorDiv(diff_0,2))], [tf.floorDiv(diff_1,2), tf.sub(diff_1, tf.floorDiv(diff_1,2))], [0, 0]], 'CONSTANT')
    if (img.shape[0] != 160 || img.shape[1] != 160){
        img = tf.image.resizeBilinear(img, [160, 160])
    }

    img = tf.expandDims(img, 0)
    img = tf.div(img, 127.5)
    img = tf.sub(img, 1)

    embedding = await model.predict(img)
    return embedding.arraySync()[0]
}

function findEuclideanDistance(source_representation, test_representation){
    // if (typeof source_representation == 'object'){
    //     source_representation = tf.tensor(source_representation)
    // }
    // if (typeof test_representation == 'object'){
    //     test_representation = tf.tensor(test_representation)
    // }
    var euclidean_distance = tf.sub(source_representation, test_representation)
    euclidean_distance = tf.sum(tf.mul(euclidean_distance, euclidean_distance))
    euclidean_distance = tf.sqrt(euclidean_distance)
    return euclidean_distance
}

function l2_normalize(x){
    return tf.div(x, tf.sqrt(tf.sum(tf.mul(x, x))))
}

async function run(){
    url1 = '/home/whoisltd/works/face-comparison/dat3.JPG'
    url2 = "/home/whoisltd/works/face-comparison/dat4.png"
    img1 = await mtcnn.detect(url1)
    img2 = await mtcnn.detect(url2)
    boxes1 = img1.boxes
    boxes2 = img2.boxes
    var data1 = await sharp(url1).rotate().toBuffer()
    var tensor1 = tf.node.decodeImage(data1)
    var data2 = await sharp(url2).rotate().toBuffer()
    var tensor2 = tf.node.decodeImage(data2)

    var face1 = tf.slice(tensor1, [boxes1[1], boxes1[0]], [boxes1[3]-boxes1[1], boxes1[2]-boxes1[0]])
    var face2 = tf.slice(tensor2, [boxes2[1], boxes2[0]], [boxes2[3]-boxes2[1], boxes2[2]-boxes2[0]])
    
    const a = await img_to_endcoding(face1)
    const b = await img_to_endcoding(face2)

    dist = findEuclideanDistance(l2_normalize(a), l2_normalize(b))
    console.log('Distance between two images is', dist.dataSync()[0])
    if(dist.dataSync()[0] >=1.02){
        console.log('These images are of two different people!')
    }
    else{
        console.log('These images are of the same people!')
    }
    
}

run()
