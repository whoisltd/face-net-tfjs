const tf = require('@tensorflow/tfjs-node')
const MTCNN = require('@whoisltd/mtcnn-tfjs')

const facenet_url = 'file:///home/whoisltd/works/jits-ai-backend/api/services/func_ml/face_verification/models/facenet/model.json'
const pnet_url = 'file:///home/whoisltd/works/jits-ai-backend/api/services/func_ml/face_verification/models/pnet/model.json'
const rnet_url = 'file:///home/whoisltd/works/jits-ai-backend/api/services/func_ml/face_verification/models/rnet/model.json'
const onet_url = 'file:///home/whoisltd/works/jits-ai-backend/api/services/func_ml/face_verification/models/onet/model.json'

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

class FaceNet{
    constructor(p_net, r_net, o_net, facenet){
        this.mtcnn = new MTCNN.MTCNN(p_net, r_net, o_net)
        this.model = this.load_model(facenet)
    }
    
    async load_model(url){
        var model = await tf.loadLayersModel(url)
        return model
    }
    async img_to_endcoding(image){

        var factor_0 = tf.div(160, image.shape[0])
        var factor_1 = tf.div(160, image.shape[1])

        var factor = Math.min(factor_0.dataSync()[0], factor_1.dataSync()[0])
        
        var dsize0 = tf.cast(tf.round(tf.mul(image.shape[1], factor)), 'int32')
        var dsize1 =  tf.cast(tf.round(tf.mul(image.shape[0], factor)), 'int32')

        var resized = tf.image.resizeBilinear(image, [dsize0.dataSync()[0], dsize1.dataSync()[0]])

        var diff_0 = tf.sub(160, resized.shape[0])
        var diff_1 = tf.sub(160, resized.shape[1])

        var img = tf.pad(resized, [[tf.floorDiv(diff_0,2), tf.sub(diff_0, tf.floorDiv(diff_0,2))], [tf.floorDiv(diff_1,2), tf.sub(diff_1, tf.floorDiv(diff_1,2))], [0, 0]], 'CONSTANT')
        if (img.shape[0] != 160 || img.shape[1] != 160){
            img = tf.image.resizeBilinear(img, [160, 160])
        }

        img = tf.expandDims(img, 0)
        img = tf.div(img, 127.5)
        img = tf.sub(img, 1)

        var embedding = (await this.model).predict(img)
        return embedding.arraySync()[0]
    }

    findEuclideanDistance(source_representation, test_representation){
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

    l2_normalize(x){
        return tf.div(x, tf.sqrt(tf.sum(tf.mul(x, x))))
    }

    face_distance_to_conf(face_distance, face_match_threshold){
        if (face_distance > face_match_threshold){
            var range = (1.64 - face_match_threshold)
            var linear_val = (1.64 - face_distance) / (range * 2.0)
            return linear_val
        }
        else{
            var range = face_match_threshold
            var linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * Math.pow((linear_val - 0.5) * 2, 0.2))
        }
    }

    async run(){
        var url1 = '/home/whoisltd/works/face-comparison/dat3.JPG'
        var url2 = "/home/whoisltd/works/face-comparison/ben2.jpg"

        var img1 = await this.mtcnn.crop_face(url1)
        var img2 = await this.mtcnn.crop_face(url2)
        
        const a = await this.img_to_endcoding(img1)
        const b = await this.img_to_endcoding(img2)

        var dist = this.findEuclideanDistance(this.l2_normalize(a), this.l2_normalize(b))
        console.log('Distance between two images is', dist.dataSync()[0])
        var conf = this.face_distance_to_conf(dist.dataSync()[0], 1.05)
        if(dist.dataSync()[0] >=1.05){
            console.log('These images are of two different people! Probability:', conf)
        }
        else{
            console.log('These images are of the same people! Probability:', conf)
        }
        
        
    }
}

facenet = new FaceNet(pnet_url, rnet_url, onet_url, facenet_url)
facenet.run()

// test runtime of facenet

// async function test(){
//     facenet = new FaceNet(pnet_url, rnet_url, onet_url, facenet_url)
//     console.time()
//     await facenet.run()
//     console.timeEnd()
//     console.time('Done')
//     await facenet.run()
//     console.timeEnd('Done')
// }
// test()

