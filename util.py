import tensorflow as tf
import numpy as np

model = None
output_class = ["Batteries", "Clothes", "E-wastes", "Glass", "Light Blubs", "Medical wastes", "Metals", "Organic Wastes", "Papers", "Plastics"]
data = {
    "Batteries": [
        "Battery recycling is an activity aimed at reducing the disposal of batteries as municipal solid waste. Batteries contain heavy metals and toxic chemicals, and disposing of them alongside regular household waste raises concerns about soil contamination and water pollution. Most types of batteries can be recycled, with lead-acid automotive batteries having a recycling rate of nearly 90%. Rechargeable batteries, such as nickel–cadmium (Ni-Cd), nickel metal hydride (Ni-MH), lithium-ion (Li-ion), and nickel–zinc (Ni-Zn), can also be recycled. Currently, there is no cost-effective recycling option for disposable alkaline batteries, although consumer disposal guidelines may vary by region.",
        "4XOAGNzWvqY", "oKFOqMZmuA8"
    ],
    "Clothes": [
        "Textile recycling involves recovering fiber, yarn, or fabric and reprocessing it into useful products. Textile waste is collected from various sources, sorted, and processed based on its condition, composition, and resale value. The end result can range from energy and chemical production to new clothing. Due to a trend of overconsumption in global fashion culture, textile recycling has become a focal point of sustainability efforts. The rise of 'fast fashion' has led many consumers to view clothes as disposable, necessitating the development of recycling technologies that can reduce reliance on natural resources.",
        "Bhi7S06pwv4", "IHPBJySIXZw"
    ],
    "E-wastes": [
        "Electronic waste, or e-waste, refers to discarded electrical or electronic devices. This includes used electronics destined for refurbishment, reuse, resale, salvage, recycling, or disposal. Informal processing of e-waste in developing countries can have harmful effects on human health and the environment. Components like CPUs may contain hazardous materials, including lead, cadmium, and brominated flame retardants. The rapid expansion of technology has resulted in a substantial increase in e-waste generation, creating significant disposal challenges.",
         "w0ikFMTuS9c"
    ],
    "Glass": [
        "Glass recycling is the process of converting waste glass into usable products. Crushed glass, known as cullet, is used in glass manufacturing. There are two types of cullet: internal (defective products from manufacturing) and external (waste glass collected for recycling). To recycle glass, it must be purified and cleaned of contaminants. Depending on the end use, it may also need to be sorted by color. Many recyclers collect different colors separately, as glass retains its color after recycling.",
        "bYVih298o1Y", "6R8YObQbE88"
    ],
    "Medical wastes": [
        "Medical waste encompasses any waste generated during healthcare activities, including diagnosis, treatment, or immunization. This category includes both hazardous and non-hazardous materials such as infectious waste, sharps, pharmaceuticals, and chemical waste. Improper disposal can pose serious public health risks, including the spread of infectious diseases and soil and water contamination. Specialized disposal methods, such as incineration and autoclaving, are necessary to ensure safe management. Recycling certain medical materials can help reduce environmental impact, but medical waste management is highly regulated.",
        "oGZx9ZZ-jFc", "X3JkPsfwwU0"
    ],
    "Light Bulbs": [
        "Recycling light bulbs prevents hazardous materials from entering the environment. Fluorescent bulbs contain mercury, a highly toxic heavy metal, while some HID bulbs contain radioactive substances. LED bulbs, although mercury-free, may contain nickel, lead, and trace amounts of arsenic. Light bulbs often break when discarded improperly, releasing hazardous materials. Recycling allows for the recovery of glass, metals, and other components, making it essential for environmental health.",
        "GbE9C2tTW2k", "PkfX4sZwrQ4"
    ],
    "Metals": [
        "Various metals are used extensively in industrial processes. Since the industrial revolution, metal consumption has soared due to mass production and lower prices. Aluminum is the most consumed metal globally, followed by copper, zinc, lead, and nickel. Precious metals like gold are also found in electronic devices. However, metal resources are limited, and depletion poses a significant future challenge. Effective measures, including metal recycling, are necessary to address this issue.",
        "qAGCI0-pQ3E", "rgEEXhbar3A"
    ],
    "Organic Wastes": [
        "Organic wastes consist of materials originating from living organisms and can be found in municipal solid waste, industrial solid waste, agricultural waste, and wastewater. While often disposed of in landfills or incinerators, some organic materials are biodegradable and suitable for composting. Common organic wastes include food, paper, wood, and yard waste. With landfill capacity dwindling, municipal composting sites are increasing, along with individual composting efforts.",
        "lHyL41grGUo", "2I8Tjb4Fy-Q"
    ],
    "Papers": [
        "Paper recycling involves turning waste paper into new products. This process has several benefits, including reducing landfill waste and minimizing methane emissions from decomposing paper. Recycling paper keeps carbon locked in the material, preventing it from entering the atmosphere. About two-thirds of paper products in the U.S. are recovered and recycled, although not all become new paper. After repeated processing, paper fibers become too short for further use, necessitating the addition of virgin fibers from sustainably sourced trees.",
        "jAqVxsEgWIM", "xhW0RTg8kRI"
    ],
    "Plastics": [
        "Plastic recycling is the recovery and reprocessing of scrap plastic into useful products. Despite the potential for recycling, less than 10% of plastic has been recycled due to misleading symbols on packaging and technical challenges. Materials recovery facilities sort and process plastics, but many struggle to be economically viable. The plastics industry has long been aware of the difficulties in recycling most plastics, yet continues to produce large quantities of virgin plastic.",
        "rYwBL_6hB2I", "I_fUpP-hq3A"
    ]
}



video_data = {
    "Batteries": ["tMvbg4XvM7Y", "eO-X8Gw2nXY"],
    "Clothes": ["vAm0Pd5frh4", "YvBS6qagQdE"],
    "E-wastes": ["g1Ij4Emz8XQ", "s2xrarUWVRQ"],
    "Glass": ["18oxQkP4qQ0", "6R8YObQbE88"],
    "Medical wastes": ["N1FeTxiXcgI", "EtXP9p_zJUA"],
    "Light Bulbs": ["GbE9C2tTW2k", "PkfX4sZwrQ4"],
    "Metals": ["qAGCI0-pQ3E", "rgEEXhbar3A"],
    "Organic Wastes": ["lHyL41grGUo", "2I8Tjb4Fy-Q"],
    "Papers": ["isEV-mCFPiY", "jAqVxsEgWIM"],
    "Plastics": ["rYwBL_6hB2I", "I_fUpP-hq3A"]
}


# When loading the model, specify the input layer manually


def load_artifacts():
    global model
    # Remove the 'options' argument as it is not required here.
    model = tf.keras.models.load_model("waste_model_v1_inceptionV3.h5", compile=False)



def classify_waste(image_path):
    global model, output_class
    # Load and preprocess the image
    test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  # Ensure target_size matches model's expected input
    test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255  # Normalize
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Predict the class
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]  # Get the predicted class name

    # Fetch the relevant data from the dictionary
    description = data[predicted_value][0]  # Description of the waste type
    video_ids = video_data[predicted_value]  # Get video IDs from the new dictionary

    # Construct YouTube embed links
    video_embeds = [
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
        for video_id in video_ids
    ]
    
    image_id = data[predicted_value][2]      # Associated Image ID

    return predicted_value, description, video_embeds, image_id

