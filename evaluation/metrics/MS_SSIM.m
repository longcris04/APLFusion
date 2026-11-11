function MS_SSIM = MS_SSIM(img1, img2,imgf)
MS_SSIM1 =ssim_index(img1,imgf);
MS_SSIM2 =ssim_index(img2,imgf);
MS_SSIM=(MS_SSIM1+MS_SSIM2)./2;
end