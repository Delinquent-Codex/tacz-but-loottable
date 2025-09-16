# Spawn a Warzone supply chest at the executor's location with randomized loot
setblock ~ ~ ~ chest[facing=north]
data modify block ~ ~ ~ LootTable set value "tacz:warzone/chest"
