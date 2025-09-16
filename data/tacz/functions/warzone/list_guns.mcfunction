# List all Warzone loot tables that can be rolled
execute if entity @s run tellraw @s {"text":"Available Warzone drops:","color":"gold"}
execute if entity @s run data get storage tacz:warzone guns
