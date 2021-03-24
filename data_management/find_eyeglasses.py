import pandas as pd
import zipfile


df_att = pd.read_csv(f'C:\\Users\\EGimenez\\ME\\projects\\BGSE\\TFM\\Anno-20210225T181807Z-001\\Anno\\list_attr_celeba_3.txt', sep=' ')
df_id = pd.read_csv(f'C:\\Users\\EGimenez\\ME\\projects\\BGSE\\TFM\\Anno-20210225T181807Z-001\\Anno\\identity_CelebA.txt', sep=' ', header =None, names=['img_id','person_id'])
result = pd.merge(df_att, df_id, how="left", on=['img_id', 'img_id'])
result.pop('Unnamed: 41')
result = result.groupby('person_id')[['Eyeglasses']].var()
result = result.sort_values(by=['Eyeglasses'], ascending=False)

my_people = df_id[df_id['person_id']==4958]

with zipfile.ZipFile(f'C:\\Users\\EGimenez\\ME\\projects\\BGSE\\TFM\\img_align_celeba.zip') as z:
    with open(f'C:\\Users\\EGimenez\\ME\\projects\\BGSE\\TFM\\code\\glow-master\\demo\\test\\082099.jpg', 'wb') as f:
        f.write(z.read('img_align_celeba/082099.jpg'))