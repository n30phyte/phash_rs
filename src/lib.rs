use image::imageops::FilterType;
use image::{io::Reader as ImageReader, DynamicImage};
use libc::c_char;
use ndarray::{s, Array2};
use nshare::ToNdarray2;
use rustdct::DctPlanner;
use std::{ffi::CStr, path::Path};

pub struct Image {
    image: DynamicImage,
    scaled_image: Option<DynamicImage>,
    greyscale_image: Option<DynamicImage>,
    dct_matrix: Option<Array2<f32>>,
}

fn vec_to_u64(slice: &[bool]) -> u64 {
    slice
        .iter()
        .fold(0, |acc, &b| acc * 2 + if b { 1 } else { 0 } as u64)
}

fn vec_median(slice: &[f32]) -> f32 {
    let mut numbers = slice.to_vec();

    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = (numbers.len() - 1) / 2;
    if numbers.len() % 2 == 0 {
        (numbers[mid] + numbers[mid + 1]) / 2.0
    } else {
        numbers[mid]
    }
}

impl Image {
    pub fn new<P: AsRef<Path>>(path: P) -> Option<Image> {
        match ImageReader::open(path) {
            Ok(reader) => match reader.decode() {
                Ok(image) => Some(Image {
                    image,
                    scaled_image: None,
                    greyscale_image: None,
                    dct_matrix: None,
                }),
                Err(_) => None,
            },
            Err(_) => None,
        }
    }

    fn convert_greyscale(&mut self) {
        self.greyscale_image = Some(self.image.grayscale());
    }

    fn scale_image(&mut self) {
        const TARGET_SIZE: u32 = 32; // 32x32 target
        let greyscale_image = self.greyscale_image.take().unwrap();

        let scaled_image = greyscale_image.resize(TARGET_SIZE, TARGET_SIZE, FilterType::Triangle);
        self.scaled_image = Some(scaled_image);
    }

    fn compute_dct(&mut self) {
        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(32);

        let greyscale_buffer = self.scaled_image.take().unwrap();
        let luma_buffer = greyscale_buffer
            .as_luma8()
            .take()
            .unwrap()
            .clone()
            .into_ndarray2();

        let mut data = Vec::new();

        for row in luma_buffer.rows() {
            let mut row_vec: Vec<f32> = row.iter().map(|num| *num as f32).collect();
            dct.process_dct2(&mut row_vec);
            data.extend_from_slice(&row_vec[..]);
        }

        let intermediate_matrix = Array2::from_shape_vec((32, 32), data).unwrap();

        let transposed = intermediate_matrix.reversed_axes();

        data = Vec::new();

        for row in transposed.rows() {
            let mut row_vec: Vec<f32> = row.iter().map(|num| *num as f32).collect();
            dct.process_dct2(&mut row_vec);
            data.extend_from_slice(&row_vec[..]);
        }

        let final_matrix = Array2::from_shape_vec((32, 32), data).unwrap();
        let transposed = final_matrix.reversed_axes();
        self.dct_matrix = Some(transposed);
    }

    fn construct_hash(&mut self) -> Vec<bool> {
        let matrix = self.dct_matrix.take().unwrap();
        let top_left = matrix.slice(s![0..8, 0..8]);
        let flattened: Vec<f32> = top_left.iter().map(|num| *num).collect();
        let med = vec_median(&flattened);

        flattened.iter().map(|num| *num > med).collect()
    }

    pub fn calculate_hash(&mut self) -> u64 {
        self.convert_greyscale();
        self.scale_image();
        self.compute_dct();

        let hash = self.construct_hash();
        vec_to_u64(&hash)
    }
}

#[no_mangle]
pub extern "C" fn calculate_hash(file_path: *const c_char) -> u64 {
    let c_str = unsafe {
        assert!(!file_path.is_null());
        CStr::from_ptr(file_path)
    };

    let file_path = c_str.to_str().unwrap();
    let image = Image::new(file_path);
    if let Some(mut img) = image {
        img.calculate_hash()
    } else {
        return 0;
    }
}
