use std::{
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use opus::{Application, Encoder};

fn main() -> anyhow::Result<()> {
    let audio_data = Arc::new(Mutex::new(Vec::<f32>::new()));
    let audio_data_c = audio_data.clone();

    let host = cpal::default_host();

    // Get the default *output* device (speaker)
    let device = host
        .default_output_device()
        .expect("no output device available");

    // On Windows, output devices can be used in loopback mode to capture their playback.
    let config = device.default_output_config()?.config();

    println!("{:?}", config);

    // Create an Opus encoder
    let mut encoder = Encoder::new(
        config.sample_rate.0 as u32,
        opus::Channels::Mono,
        Application::Audio,
    )?;

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            // Encode float samples to Opus

            // let _encoded_len = encoder.encode_float(data, &mut opus_buf).unwrap();
            // Write to file, socket, etc.

            audio_data.lock().unwrap().extend_from_slice(data);
        },
        |err| eprintln!("Stream error: {err}"),
        None,
    )?;

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(10));
    stream.stop()?;

    // let _encoded_len = encoder.encode_float(data, &mut opus_buf).unwrap();

    let mut opus_buf = [0u8; 4000];
    let mut file = File::create("output.opus")?;

    for chunk in audio_data_c.lock().unwrap().chunks(960) {
        // 20 ms chunks at 48 kHz
        let len = encoder.encode_float(chunk, &mut opus_buf)?;
        file.write_all(&opus_buf[..len])?;
    }

    Ok(())
}
