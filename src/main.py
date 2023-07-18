import os
from txt2img import txt2img

if True:
    image = txt2img(
        checkpoint = "aZovyaRPGArtistTools_sd21768V1",
        positive = """old man <lora:DesuZenkaiV21beta2-lyco:1.4>""",
        negative = """""",
        steps = 10,
        seed = 3125748766
    )

    image.save("../output/sd2.png")

prompts = {
    "hada": """old man <lora:KyoAni:1.5>""", 
    "ia3": """old man <lora:Cammy-400:1>""", 
    "lokr": """old man <lora:mikapikazo:1>""", 
    "full": """old man <lora:test_full:1>""",
    "lora": """old man <lora:animeScreencapStyle_v230epochs:1>""",
}
i = 0
for type, prompt in prompts.items():    
    image = txt2img(
        checkpoint = "juggernaut_final",
        positive = prompt,
        negative = """""",
        steps = 10,
        seed = 3125748766
    )

    image.save("../output/" + type + ".png")
    i += 1
