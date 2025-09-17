# List all Warzone loot tables that can be rolled
execute if entity @s run tellraw @s {"text":"Warzone drop tables loaded:","color":"gold"}
execute if entity @s run data get storage tacz:warzone guns
execute if entity @s run tellraw @s {"text":"Common variants:","color":"gray"}
execute if entity @s run data get storage tacz:warzone guns_common
execute if entity @s run tellraw @s {"text":"Rare variants:","color":"aqua"}
execute if entity @s run data get storage tacz:warzone guns_rare
execute if entity @s run tellraw @s {"text":"Legendary variants:","color":"gold"}
execute if entity @s run data get storage tacz:warzone guns_legendary
execute if entity @s run tellraw @s {"text":"Use /function tacz:warzone/give_drop_* or /loot give @s loot tacz:warzone/chest_* to target a tier.","color":"yellow"}
