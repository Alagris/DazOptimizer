# How to install 

Install blender, DAZ and Diffeomorphic.
Then open blender go to Scripting tab, create a new file paste there the contents of `__init__.py` and then click the arrow button to run the script.
You should now see `DazOptim` appear right bar in the 3d view.
Go to Window>Toggle System Console and look for a line more or less like this (should be somewhere on top. It appears when you execute the script)
```bash
&some/path/to/python.exe  -m pip install Pillow opencv-python
```
Copy it, open powershell with as an administrator, paste the command and execute it. 

# How to use

Just execute the steps in the order as they appear. Here is what each of them does:

- Delete default cube - it's recommended to start with a clean scene.
- Load Daz 
  - you should save a .duf scene with your DAZ character and also .dbz just like you would do it using diffeomorphic.
  - then instead of loading .dbz file using diffeomorphic use DazOptim. It will set up the import procedure with all the right configurations. You just need to find the .dbz file and load it. 
- Save blend file
  - this is a requirement for the next step to work
- Save textures
  - saves all the textures into a directory next to the .blend file
- Merge all rigs
  - Initially every body part has it's own rig, potentially with additional bones. This will merge all rigs into one 
- Save male fav morphs
  - Use this if you intend to load morph targets. If your character is female, then use the next button instead. 
- Save female fav morphs
  - Use this instead of the button before if your character is female
- Load fav morphs
  - Loads all the morphs targets saved into the generated JSON file. If you have some custom morph targets that you'd like to load you can add them to `MORPHS` global variable in `blender_export_script.py`
- Transfer morphs to geografts
  - If you loaded any morphs, use this step to transfer them to all of your geografts. 
  - You should make sure that your geografts are added to the list in `GEOGRAFTS` global variable.
- Subdivide breast bones
  - DAZ characters have a single bone for each breast. If you want to have better jiggle physics, you can subdivide those bones (by default into 3 on each side).  
- Add glute bones
  - DAZ characters have no glute bones. This will add them and auto-generate the weights. Use it if you want to have ass jiggle physics
- Add thigh bones
  - Analogical as with glute bones  
- Simplify materials
  - DAZ materials are quite complex and they don't translate very well to unreal. This step will simplify them to a single `Principled BSDF` node with a bunch of `Image Textures` for color, roughness and normals. This operation is lossy and will alter the color of your character. You should later restore this color in the new materials in unreal engine. 
- Optimize eyes mesh
  - DAZ eyes are very heavy to render. This operation will remove redundant geometry.
- Optimize eyelashes
  - DAZ eyelashes consist of three layers of transparent hair cards. This will reduce them to a single layer and replace the texture with one that has all three layers combined into one. This texture is called `eyebrows_and_eyelashes.png` and can be found in the repo. You should copy it and place either next to the .blend file or in its parent directory.  
- Optimize eyebrows
  - DAZ has strand-based eyebrows. This will generate cards instead. Again you need to have `eyebrows_and_eyelashes.png`. 
- remove old eyebrows
  - This will remove the original strand-based eyebrows. It is a separate step because you might want to manually readjust the card-based eyebrows and compare them with the original ones while doing so. 
- Apply eyebrows
  - Once you're done adjusting the auto-generated card-based eyebrows, click this button to apply all modifiers. Usually the auto-generated eyebrows should be positioned perfectly and you don't need any manual adjustments
- Transfer FACS to Eyebrows
  - If you're using FACS (morph for facial expressions) you probably want the new eyelashes and eyebrows to follow those morphs. Use this button to transfer FACS to eyebrows.
- Simplify eyes material
  - In essence, this step does the same thing as "Simplify materials" but eye material is a little more complex than the rest of body materials and requires its own treatment  
- Separate iris UVs
  - If you do this, you will be later able to independently adjust iris color in unreal, without affecting the color of the rest of the eyes.
- Optimize golden palace UVs
  - If you're using golden palace, use this step to generate a new UV map that is "game ready" (originally golden palace might have several UV maps and then use complicated material logic to blend them together).
- golden palace prepare baking
  - Because golden palace uses very complex materials, it is not possible to just "simplify" it like we did for eyes and rest of body. Instead, we will have to bake those materials into a single texture. If you click this button, the plugin will modify golden palace materials by adding a new image texture that we will bake to. It will also set up some configuration necessary for baking.
- select golden palace color for baking
  - This configures blender for diffuse color baking
- Bake
  - it's the same as just clicking the Bake button in vanilla blender. This step is here just for convenience. Also, remember to go to Edit>Preferences>System and use CUDA/OptiX (if you have). This will speed up baking a lot. 
- select golden palace normals for baking
  - This configures blender for normal baking
- Bake
  - same as before
- select golden palace roughness for baking
  - This configures blender for roughness baking
- Bake
  - same as before
- use principled bsdf
  - you don't need to use this button unless a more advanced user and want to experiment with different ways of baking.
- use diffuse bsdf
  - you don't need to use this button unless a more advanced user and want to experiment with different ways of baking.
- Save baked golden palace textures
  - Saves the baked textures to a file. This file will later be used in the "Merge textures" step. Also, if in the future you decide to redo the whole process but you're using the same skin material, you can skip the whole process of baking golden palace (no need to do the "golden palace prepare baking" step) and jump straight to "Merge Geografts" steps.  
- Merge Geografts
  - This will merge all geografts into a isngle texture. The only exception are penis geografts listed in `DICK_GEOGRAFTS` global variable. In unreal engine you would rather have those are separate meshes so that you can hide the penis when the player is wearing some clothes. If you really insist on merging the penis, move your graft from `DICK_GEOGRAFTS` to `MALE_ONLY_GEOGRAFTS` instead.
- Simplify golden palace materials
  - Now that you saved all of the baked golden palace textures into a file, it is possible to simplify the material just like we did for other body parts before. This step will also switch to using the new optimized golden palace UVs.
- Merge eyes
  - Do this if you want the eyes to be part of the body mesh
- Merge mouth
 - Do this if you want the mouth to be part of the body mesh
- Remove tear
  - This is not necessary in a game
- Merge eyebrows+eyelashes
  - Do this if you want eyebrows and eyelashes to be a single mesh with a single (optimized) material.
- Merge textures
  - This step will now scan all of our simplified materials, extract their textures, and concatenate all the textures into a single texture atlas. This atlas will be saved next to the .blend file. However at this point this atlas is not yet used by any of the materials.
  - If in the future you decide to redo the whole process but you still use the same skin materials, you can skip this step. There would be no need to regenerate the same atlas twice. You can even generate one atlas for each skin and then ad-hoc swap them in the game. The UVs are always going to be compatible. 
- Optimize UVs
  - This will reshuffle all UVs to match the texture atlas generated previously. However, I recommend using the "half GP" instead as the baking process is not perfect and a seam around golden palace might sometimes be noticeable.
  - At this point you might notice that your character's material got all messed up. This is expected and will be fixed in "Unify skin materials into one" step.
- Optimize UVs (half GP)
  - This is almost identical to the previous step except that only the clit, labias, vagina and anus will use the baked golden palace textures. The skin of the pubic mound will keep using the original textures. This will always produce a seamless material.
  - At this point you might notice that your character's material got all messed up. This is expected and will be fixed in "Unify skin materials into one" step.
- Separate Lip UVs
  - By using this you will be able to independently adjust color of the lips without affecting the skin around it. Very efficient in a game engine.
- Unify skin materials into one
  - This will remove all previous materials and create a new single unified material that uses the texture atlas instead of all the individual original textures. It also uses the new optimizes UVs.
- Merge eyelashes+body
  - Use this if you want facial hair to be part of the body mesh. If you don't do this, then facial mesh won't respond to morph targets in unreal. You will have to export facial hair separately instead and apply morphs separately to them as well.  
- Fit skin-tight clothes
  - If you have any skin-tight clothes, add them to the `CLOTHES` list and run this step. It will use shrinkwrap modifier to fix any potential issues with clothes intersecting the body mesh. 
- Transfer new bones to clothes
  - If you added breast, glute or thigh bones you can use this to transfer their weights to the clocthes. Make sure you add your clothes to the `CLOTHES` global variable
- Apply skin-tight clothes
  - Applies all modifiers for the skin-tight clothes
- Optimize hair
  - TODO
- Compare to UE5 Skeleton
  - Use this step to see how much your skeleton differs from UE5 Manny (results are printed in the system console window).
- Convert to UE5 Skeleton
  - Renames most bones and reparents/moves some of them. 
- Scale to quinn height
  - Use this if you want your character to match UE5 Quinn in proportions.
- Reorient bones
  - Use this step to change the bone tails to follow the unreal's convention. 
- Add UE5 IK bones
  - Add the missing IK bones that UE5 Manny uses.
- Scale to ue5 units
  - This is a very important step. It changes the units to match those in unreal.
- Export to fbx
  - This will generate an .fbx file right next to the .blend file
