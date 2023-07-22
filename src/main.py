import hacks
import os
from txt2img import txt2img

image = txt2img(
    checkpoint = "juggernaut_final",
    positive = """(old man:1.2) <lora:KyoAni:1>""",
    negative = """""",
    steps = 20,
    seed = 3125748766
)

image.save("../output/output.png")

if False:
    prompts = {
        "hada": """old man <lora:KyoAni:1.5>""", 
        "ia3": """old man <lora:Cammy-400:1>""", 
        "lokr": """old man <lora:mikapikazo:1>""", 
        "full": """old man <lora:test_full:1>""",
        "lora": """old man <lora:animeScreencapStyle_v230epochs:1>""",
        "all": """
            old man
            <lora:KyoAni:1.5>
            <lora:Cammy-400:1>
            <lora:mikapikazo:1>
            <lora:test_full:1>
            <lora:animeScreencapStyle_v230epochs:1>
        """
    }

    for type, prompt in prompts.items():    
        image = txt2img(
            checkpoint = "juggernaut_final",
            positive = prompt,
            negative = """""",
            steps = 10,
            seed = 3125748766
        )

        image.save("../output/" + type + ".png")
