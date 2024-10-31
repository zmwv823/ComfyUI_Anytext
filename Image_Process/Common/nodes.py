from ...UL_common.common import tensor2numpy_cv2, pil2tensor, cv2img_canny
    
class UL_Image_Process_Common_Cv2_Canny:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            "low_threshold": ("INT", {"default": 64,"min": 0, "max": 500, "step": 1, "tooltip": "测试"}),
            "high_threshold": ("INT", {"default": 100,"min": 0, "max": 500, "step": 1, "tooltip": "测试"}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "UL_Image_Process_Common_Cv2_Canny"
    CATEGORY = "UL Group/Image Process Common"
    TITLE = "Common Cv2 Canny"
        
    def UL_Image_Process_Common_Cv2_Canny(self, image, low_threshold, high_threshold):
        cv2_img = tensor2numpy_cv2(image)
        cv2_canny_img = cv2img_canny(cv2_img, low_threshold, high_threshold)
        image = pil2tensor(cv2_canny_img)
        return (image, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UL_Image_Process_Common_Cv2_Canny": UL_Image_Process_Common_Cv2_Canny,
}